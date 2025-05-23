; a6xx microcode
; Version: 016ee001

[016ee001]
[#jumptbl]
mov $01, 0x830	; CP_SQE_INSTR_BASE
mov $02, 0x2
cwrite $01, [$00 + @REG_READ_ADDR]
cwrite $02, [$00 + @REG_READ_DWORDS]
mov $01, $regdata
mov $02, $regdata
add $01, $01, 0x4
addhi $02, $02, 0x0
mov $03, 0x1
cwrite $01, [$00 + @MEM_READ_ADDR]
cwrite $02, [$00 + @MEM_READ_ADDR+0x1]
cwrite $03, [$00 + @MEM_READ_DWORDS]
rot $04, $memdata, 0x8
ushr $04, $04, 0x6
sub $04, $04, 0x4
add $01, $01, $04
addhi $02, $02, 0x0
mov $rem, 0x80
cwrite $01, [$00 + @MEM_READ_ADDR]
cwrite $02, [$00 + @MEM_READ_ADDR+0x1]
cwrite $02, [$00 + @LOAD_STORE_HI]
cwrite $rem, [$00 + @MEM_READ_DWORDS]
cwrite $00, [$00 + @PACKET_TABLE_WRITE_ADDR]
(rep)cwrite $memdata, [$00 + @PACKET_TABLE_WRITE]
mov $02, 0x883	; CP_SCRATCH[0].REG
mov $03, 0xbeef
mov $04, 0xdead << 16
or $03, $03, $04
cwrite $02, [$00 + @REG_WRITE_ADDR]
cwrite $03, [$00 + @REG_WRITE]
waitin
mov $01, $data

CP_ME_INIT:
mov $02, 0x2
waitin
mov $01, $data

CP_MEM_WRITE:
mov $addr, 0xa0 << 24	; |NRT_ADDR
mov $02, 0x4
(xmov1)add $data, $02, $data
mov $addr, 0xa204 << 16	; |NRT_DATA
(rep)(xmov3)mov $data, $data
waitin
mov $01, $data

CP_SCRATCH_WRITE:
mov $02, 0xff
(rep)cwrite $data, [$02 + 0x1]!
waitin
mov $01, $data

CP_SET_DRAW_STATE:
(rep)(sds2)cwrite $data, [$00 + @DRAW_STATE_SET_HDR]
waitin
mov $01, $data

CP_SET_BIN_DATA5:
sread $02, [$00 + %SP]
swrite $02, [$00 + %SP]
mov $02, 0x7
(rep)swrite $data, [$02 + 0x1]!
waitin
mov $01, $data

CP_SET_SECURE_MODE:
mov $02, $data
setsecure $02, #l61

fxn59:
l59:
jump #l59
nop
l61:
waitin
mov $01, $data

fxn63:
l63:
cmp $04, $02, $03
breq $04, b0, #l70
brne $04, b1, #l68
breq $04, b2, #l63
sub $03, $03, $02
l68:
jump #l63
sub $02, $02, $03
l70:
ret
nop

CP_REG_RMW:
cwrite $data, [$00 + @REG_READ_ADDR]
add $02, $regdata, 0x42
addhi $03, $00, $regdata
sub $02, $02, $regdata
call #fxn63
subhi $03, $03, $regdata
and $02, $02, $regdata
or $02, $02, 0x1
xor $02, $02, 0x1
not $02, $02
shl $02, $02, $regdata
ushr $02, $02, $regdata
ishr $02, $02, $regdata
rot $02, $02, $regdata
min $02, $02, $regdata
max $02, $02, $regdata
mul8 $02, $02, $regdata
msb $02, $02
mov $usraddr, $data
mov $data, $02
waitin
mov $01, $data

CP_MEMCPY:
mov $02, $data
mov $03, $data
mov $04, $data
mov $05, $data
mov $06, $data
l99:
breq $06, 0x0, #l105
cwrite $03, [$00 + @LOAD_STORE_HI]
load $07, [$02 + 0x4]!
cwrite $05, [$00 + @LOAD_STORE_HI]
jump #l99
store $07, [$04 + 0x4]!
l105:
waitin
mov $01, $data

CP_MEM_TO_MEM:
cwrite $data, [$00 + @MEM_READ_ADDR]
cwrite $data, [$00 + @MEM_READ_ADDR+0x1]
mov $02, $data
cwrite $data, [$00 + @LOAD_STORE_HI]
mov $rem, $data
cwrite $rem, [$00 + @MEM_READ_DWORDS]
(rep)store $memdata, [$02 + 0x4]!
waitin
mov $01, $data

IN_PREEMPT:
cread $02, [$00 + 0x101]
brne $02, 0x1, #l125
nop
bl #fxn59
nop
nop
nop
waitin
mov $01, $data
l125:
iret
nop

CP_BLIT:
CP_BOOTSTRAP_UCODE:
CP_COND_EXEC:
CP_COND_REG_EXEC:
CP_COND_WRITE5:
CP_CONTEXT_REG_BUNCH:
CP_CONTEXT_SWITCH:
CP_CONTEXT_SWITCH_YIELD:
CP_CONTEXT_UPDATE:
CP_DRAW_AUTO:
CP_DRAW_INDIRECT:
CP_DRAW_INDIRECT_MULTI:
CP_DRAW_INDX:
CP_DRAW_INDX_INDIRECT:
CP_DRAW_INDX_OFFSET:
CP_DRAW_PRED_ENABLE_GLOBAL:
CP_DRAW_PRED_ENABLE_LOCAL:
CP_DRAW_PRED_SET:
CP_END_BIN:
CP_EVENT_WRITE:
CP_EVENT_WRITE_CFL:
CP_EVENT_WRITE_SHD:
CP_EVENT_WRITE_ZPD:
CP_EXEC_CS:
CP_EXEC_CS_INDIRECT:
CP_IM_LOAD:
CP_IM_LOAD_IMMEDIATE:
CP_INDIRECT_BUFFER:
CP_INDIRECT_BUFFER_CHAIN:
CP_INDIRECT_BUFFER_PFD:
CP_INTERRUPT:
CP_INVALIDATE_STATE:
CP_LOAD_STATE6:
CP_LOAD_STATE6_FRAG:
CP_LOAD_STATE6_GEOM:
CP_MEM_TO_REG:
CP_MEM_WRITE_CNTR:
CP_NOP:
CP_PREEMPT_DISABLE:
CP_RECORD_PFP_TIMESTAMP:
CP_REG_TEST:
CP_REG_TO_MEM:
CP_REG_TO_MEM_OFFSET_MEM:
CP_REG_TO_MEM_OFFSET_REG:
CP_REG_TO_SCRATCH:
CP_REG_WRITE:
CP_REG_WR_NO_CTXT:
CP_RUN_OPENCL:
CP_SCRATCH_TO_REG:
CP_SET_AMBLE:
CP_SET_BIN_DATA5_OFFSET:
CP_SET_DRAW_INIT_FLAGS:
CP_SET_MARKER:
CP_SET_MODE:
CP_SET_PROTECTED_MODE:
CP_SET_PSEUDO_REG:
CP_SET_STATE:
CP_SET_SUBDRAW_SIZE:
CP_SET_VISIBILITY_OVERRIDE:
CP_SKIP_IB2_ENABLE_GLOBAL:
CP_SKIP_IB2_ENABLE_LOCAL:
CP_SMMU_TABLE_UPDATE:
CP_START_BIN:
CP_TEST_TWO_MEMS:
CP_WAIT_FOR_IDLE:
CP_WAIT_FOR_ME:
CP_WAIT_MEM_GTE:
CP_WAIT_MEM_WRITES:
CP_WAIT_REG_EQ:
CP_WAIT_REG_MEM:
CP_WAIT_TWO_REGS:
CP_WHERE_AM_I:
IN_GMU_INTERRUPT:
IN_IB_END:
PKT4:
UNKN0:
UNKN1:
UNKN103:
UNKN104:
UNKN105:
UNKN106:
UNKN110:
UNKN118:
UNKN119:
UNKN12:
UNKN121:
UNKN122:
UNKN123:
UNKN124:
UNKN125:
UNKN126:
UNKN127:
UNKN13:
UNKN14:
UNKN2:
UNKN21:
UNKN22:
UNKN23:
UNKN24:
UNKN27:
UNKN28:
UNKN3:
UNKN30:
UNKN31:
UNKN32:
UNKN45:
UNKN48:
UNKN5:
UNKN58:
UNKN6:
UNKN7:
UNKN73:
UNKN8:
UNKN9:
UNKN90:
UNKN93:
UNKN96:
UNKN97:
waitin
mov $01, $data
jumptbl:
.jumptbl
