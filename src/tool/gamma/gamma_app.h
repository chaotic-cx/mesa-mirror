/*
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <memory>
#include <utility>

#include "vulkan/vulkan_core.h"
#include <SDL3/SDL.h>

#include "backends/imgui_impl_vulkan.h"

#include "gamma_device.h"
#include "gamma_file_view.h"
#include "gamma_util.h"

#define GAMMA_MAX_OPEN_VIEWS 4096

#define GAMMA_CUBE_VERTEX_COUNT        (2 * 4 * 2 + 4 * 2)
#define GAMMA_FILLED_CUBE_VERTEX_COUNT (3 * 2 * 6)

struct gamma_app : gamma_device {
   SDL_Window *window;
   ImGui_ImplVulkanH_Window imgui_window;
   std::vector<std::unique_ptr<gamma_file_view>> open_files;

   PFN_vkCmdDrawMultiEXT vkCmdDrawMultiEXT;

   VkDescriptorSetLayout renderer_set_layout;
   VkPipelineLayout renderer_pipeline_layout;
   VkPipeline fill_pipeline;
   VkPipeline wireframe_pipeline;
   VkPipeline thick_wireframe_pipeline;
   VkPipeline lines_pipeline;
   VkPipeline thick_lines_pipeline;

   uint32_t sample_count;
   uint32_t max_multi_draw_count;

   std::shared_ptr<gamma_backed_buffer> cube_vertex_buffer;
   std::shared_ptr<gamma_backed_buffer> filled_cube_vertex_buffer;

   std::chrono::high_resolution_clock::time_point last_run_time;
   bool last_run_time_valid = false;
   float t = 0;
   float dt;

   gamma_vec3 selection_color;
};

int gamma_app_init(gamma_app *app);

void gamma_app_finish(gamma_app *app);

void gamma_app_resize(gamma_app *app, uint32_t width, uint32_t height);

bool gamma_app_run(gamma_app *app);
