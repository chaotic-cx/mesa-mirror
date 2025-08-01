use crate::api::icd::*;
use crate::api::types::*;
use crate::api::util::*;
use crate::core::context::*;
use crate::core::event::*;
use crate::core::queue::*;

use rusticl_opencl_gen::*;
use rusticl_proc_macros::cl_entrypoint;
use rusticl_proc_macros::cl_info_entrypoint;

use std::ptr;
use std::sync::Arc;
use std::sync::Weak;

#[cl_info_entrypoint(clGetEventInfo)]
unsafe impl CLInfo<cl_event_info> for cl_event {
    fn query(&self, q: cl_event_info, v: CLInfoValue) -> CLResult<CLInfoRes> {
        let event = Event::ref_from_raw(*self)?;
        match *q {
            CL_EVENT_COMMAND_EXECUTION_STATUS => v.write::<cl_int>(event.status()),
            CL_EVENT_CONTEXT => {
                // Note we use as_ptr here which doesn't increase the reference count.
                let ptr = Arc::as_ptr(&event.context);
                v.write::<cl_context>(cl_context::from_ptr(ptr))
            }
            CL_EVENT_COMMAND_QUEUE => {
                let ptr = match event.queue.as_ref() {
                    // Note we use as_ptr here which doesn't increase the reference count.
                    Some(queue) => Weak::as_ptr(queue),
                    None => ptr::null_mut(),
                };
                v.write::<cl_command_queue>(cl_command_queue::from_ptr(ptr))
            }
            CL_EVENT_REFERENCE_COUNT => v.write::<cl_uint>(Event::refcnt(*self)?),
            CL_EVENT_COMMAND_TYPE => v.write::<cl_command_type>(event.cmd_type),
            _ => Err(CL_INVALID_VALUE),
        }
    }
}

#[cl_info_entrypoint(clGetEventProfilingInfo)]
unsafe impl CLInfo<cl_profiling_info> for cl_event {
    fn query(&self, q: cl_profiling_info, v: CLInfoValue) -> CLResult<CLInfoRes> {
        let event = Event::ref_from_raw(*self)?;
        if event.cmd_type == CL_COMMAND_USER {
            // CL_PROFILING_INFO_NOT_AVAILABLE [...] if event is a user event object.
            return Err(CL_PROFILING_INFO_NOT_AVAILABLE);
        }

        match *q {
            CL_PROFILING_COMMAND_QUEUED => v.write::<cl_ulong>(event.get_time(EventTimes::Queued)),
            CL_PROFILING_COMMAND_SUBMIT => v.write::<cl_ulong>(event.get_time(EventTimes::Submit)),
            CL_PROFILING_COMMAND_START => v.write::<cl_ulong>(event.get_time(EventTimes::Start)),
            CL_PROFILING_COMMAND_END => v.write::<cl_ulong>(event.get_time(EventTimes::End)),
            // For now, we treat Complete the same as End
            CL_PROFILING_COMMAND_COMPLETE => v.write::<cl_ulong>(event.get_time(EventTimes::End)),
            _ => Err(CL_INVALID_VALUE),
        }
    }
}

#[cl_entrypoint(clCreateUserEvent)]
fn create_user_event(context: cl_context) -> CLResult<cl_event> {
    let c = Context::arc_from_raw(context)?;
    Ok(Event::new_user(c).into_cl())
}

#[cl_entrypoint(clRetainEvent)]
fn retain_event(event: cl_event) -> CLResult<()> {
    Event::retain(event)
}

#[cl_entrypoint(clReleaseEvent)]
fn release_event(event: cl_event) -> CLResult<()> {
    Event::release(event)
}

#[cl_entrypoint(clWaitForEvents)]
fn wait_for_events(num_events: cl_uint, event_list: *const cl_event) -> CLResult<()> {
    let evs = Event::arcs_from_arr(event_list, num_events)?;

    if let Some((first, rest)) = evs.split_first() {
        // > CL_INVALID_CONTEXT if events specified in event_list do not belong
        // > to the same context.
        if rest.iter().any(|e| e.context != first.context) {
            return Err(CL_INVALID_CONTEXT);
        }
    } else {
        // > CL_INVALID_VALUE if num_events is zero or event_list is NULL.
        return Err(CL_INVALID_VALUE);
    }

    // find all queues we have to flush
    for q in Event::deep_unflushed_queues(&evs) {
        q.flush(false)?;
    }

    // now wait on all events and check if we got any errors
    let mut err = false;
    for e in evs {
        err |= e.wait() < 0;
    }

    // CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST if the execution status of any of the events
    // in event_list is a negative integer value.
    if err {
        return Err(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
    }

    Ok(())
}

#[cl_entrypoint(clSetEventCallback)]
fn set_event_callback(
    event: cl_event,
    command_exec_callback_type: cl_int,
    pfn_event_notify: Option<FuncEventCB>,
    user_data: *mut ::std::os::raw::c_void,
) -> CLResult<()> {
    let e = Event::ref_from_raw(event)?;

    // CL_INVALID_VALUE [...] if command_exec_callback_type is not CL_SUBMITTED, CL_RUNNING, or CL_COMPLETE.
    if ![CL_SUBMITTED, CL_RUNNING, CL_COMPLETE].contains(&(command_exec_callback_type as cl_uint)) {
        return Err(CL_INVALID_VALUE);
    }

    // SAFETY: The requirements on `EventCB::new` match the requirements
    // imposed by the OpenCL specification. It is the caller's duty to uphold them.
    let cb = unsafe { EventCB::new(pfn_event_notify, user_data)? };

    e.add_cb(command_exec_callback_type, cb);

    Ok(())
}

#[cl_entrypoint(clSetUserEventStatus)]
fn set_user_event_status(event: cl_event, execution_status: cl_int) -> CLResult<()> {
    let e = Event::ref_from_raw(event)?;

    // CL_INVALID_VALUE if the execution_status is not CL_COMPLETE or a negative integer value.
    if execution_status != CL_COMPLETE as cl_int && execution_status > 0 {
        return Err(CL_INVALID_VALUE);
    }

    // CL_INVALID_OPERATION if the execution_status for event has already been changed by a
    // previous call to clSetUserEventStatus.
    if e.status() != CL_SUBMITTED as cl_int {
        return Err(CL_INVALID_OPERATION);
    }

    e.set_user_status(execution_status);
    Ok(())
}

/// implements CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST when `block = true`
pub fn create_and_queue(
    q: Arc<Queue>,
    cmd_type: cl_command_type,
    deps: Vec<Arc<Event>>,
    event: *mut cl_event,
    block: bool,
    work: EventSig,
) -> CLResult<()> {
    let e = Event::new(&q, cmd_type, deps, work);
    if !event.is_null() {
        // SAFETY: we check for null and valid API use is to pass in a valid pointer
        unsafe {
            event.write(Arc::clone(&e).into_cl());
        }
    }
    if block {
        q.queue(Arc::clone(&e));
        q.flush(true)?;
        if e.deps.iter().any(|dep| dep.is_error()) {
            return Err(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        }
        // return any execution errors when blocking
        let err = e.status();
        if err < 0 {
            return Err(err);
        }
    } else {
        q.queue(e);
    }
    Ok(())
}
