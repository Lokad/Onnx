; Assembly listing for method Lokad.Onnx.Tensor`1[float]:RunFloatMatMulKernel(int,int,int,ptr,ptr,ptr,Lokad.Onnx.TensorExecutionOptions) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       55                   push     rbp
       4881ECF0000000       sub      rsp, 240
       488DAC24F0000000     lea      rbp, [rsp+0xF0]
       33C0                 xor      eax, eax
       48898548FFFFFF       mov      qword ptr [rbp-0xB8], rax
       C4413857C0           vxorps   xmm8, xmm8, xmm8
       62717E487F8550FFFFFF vmovdqu32 zmmword ptr [rbp-0xB0], zmm8
       62717E487F8590FFFFFF vmovdqu32 zmmword ptr [rbp-0x70], zmm8
       488945D0             mov      qword ptr [rbp-0x30], rax
       897DFC               mov      dword ptr [rbp-0x04], edi
       8975F8               mov      dword ptr [rbp-0x08], esi
       8955F4               mov      dword ptr [rbp-0x0C], edx
       48894DE8             mov      qword ptr [rbp-0x18], rcx
       4C8945E0             mov      qword ptr [rbp-0x20], r8
       4C894DD8             mov      qword ptr [rbp-0x28], r9
 
G_M000_IG02:                ;; offset=0x004B
       488D7D10             lea      rdi, [rbp+0x10]
       FF15D3CFF2FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       0F847D000000         je       G_M000_IG06
       488D7D10             lea      rdi, [rbp+0x10]
       FF15A9CFF2FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseIntrinsics():bool:this]
       85C0                 test     eax, eax
       7460                 je       SHORT G_M000_IG05
       837DFC01             cmp      dword ptr [rbp-0x04], 1
       7549                 jne      SHORT G_M000_IG04
       817DF400200000       cmp      dword ptr [rbp-0x0C], 0x2000
       7C2F                 jl       SHORT G_M000_IG03
       48BF700F726B947C0000 mov      rdi, 0x7C946B720F70
       E8F7FC1A7D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15CCF90500         call     [Lokad.Onnx.MathOps:mm_m1_kblocked(int,int,int,ptr,ptr,ptr)]
       E996080000           jmp      G_M000_IG70
 
G_M000_IG03:                ;; offset=0x00A9
       48BF740F726B947C0000 mov      rdi, 0x7C946B720F74
       E8C8FC1A7D           call     CORINFO_HELP_COUNTPROFILE32
       EB20                 jmp      SHORT G_M000_IG06
 
G_M000_IG04:                ;; offset=0x00BA
       48BF780F726B947C0000 mov      rdi, 0x7C946B720F78
       E8B7FC1A7D           call     CORINFO_HELP_COUNTPROFILE32
       EB0F                 jmp      SHORT G_M000_IG06
 
G_M000_IG05:                ;; offset=0x00CB
       48BF7C0F726B947C0000 mov      rdi, 0x7C946B720F7C
       E8A6FC1A7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG06:                ;; offset=0x00DA
       488D7D10             lea      rdi, [rbp+0x10]
       FF1544CFF2FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       0F8498070000         je       G_M000_IG65
       488D7D10             lea      rdi, [rbp+0x10]
       FF151ACFF2FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseIntrinsics():bool:this]
       85C0                 test     eax, eax
       0F842C020000         je       G_M000_IG23
       837DFC02             cmp      dword ptr [rbp-0x04], 2
       0F8C0E020000         jl       G_M000_IG22
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       8B4DFC               mov      ecx, dword ptr [rbp-0x04]
       8B55FC               mov      edx, dword ptr [rbp-0x04]
       C1EA1F               shr      edx, 31
       0355FC               add      edx, dword ptr [rbp-0x04]
       83E2FE               and      edx, -2
       2BCA                 sub      ecx, edx
       2BC1                 sub      eax, ecx
       8945D4               mov      dword ptr [rbp-0x2C], eax
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       898544FFFFFF         mov      dword ptr [rbp-0xBC], eax
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       B903000000           mov      ecx, 3
       99                   cdq      
       F7F9                 idiv     edx:eax, ecx
       8D0440               lea      eax, [rax+2*rax]
       8B8D44FFFFFF         mov      ecx, dword ptr [rbp-0xBC]
       2BC8                 sub      ecx, eax
       85C9                 test     ecx, ecx
       0F8500020000         jne      G_M000_IG25
       837DFC40             cmp      dword ptr [rbp-0x04], 64
       0F8CB3010000         jl       G_M000_IG21
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000004         cmp      rax, 0x4000000
       0F8F8A010000         jg       G_M000_IG20
       488D3C24             lea      rdi, [rsp]
       488D7510             lea      rsi, [rbp+0x10]
       488B0E               mov      rcx, gword ptr [rsi]
       48890C24             mov      gword ptr [rsp], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2408           mov      gword ptr [rsp+0x08], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2410           mov      gword ptr [rsp+0x10], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2418           mov      gword ptr [rsp+0x18], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
 
G_M000_IG07:                ;; offset=0x01B4
       48894C2420           mov      gword ptr [rsp+0x20], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       48A5                 movsq    
       8B45F8               mov      eax, dword ptr [rbp-0x08]
       8BF8                 mov      edi, eax
       0FAF7DF4             imul     edi, dword ptr [rbp-0x0C]
       FF156EF80500         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488945C8             mov      gword ptr [rbp-0x38], rax
 
G_M000_IG08:                ;; offset=0x01D6
       488B45C8             mov      rax, gword ptr [rbp-0x38]
       488945B8             mov      gword ptr [rbp-0x48], rax
       48837DC800           cmp      gword ptr [rbp-0x38], 0
       7419                 je       SHORT G_M000_IG10
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       83780800             cmp      dword ptr [rax+0x08], 0
       7545                 jne      SHORT G_M000_IG14
 
G_M000_IG09:                ;; offset=0x01EF
       48BF800F726B947C0000 mov      rdi, 0x7C946B720F80
       E882FB1A7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG10:                ;; offset=0x01FE
       33C0                 xor      eax, eax
       488945C0             mov      qword ptr [rbp-0x40], rax
       EB61                 jmp      SHORT G_M000_IG15
 
G_M000_IG11:                ;; offset=0x0206
       48BF840F726B947C0000 mov      rdi, 0x7C946B720F84
       E86BFB1A7D           call     CORINFO_HELP_COUNTPROFILE32
       E9A8000000           jmp      G_M000_IG17
 
G_M000_IG12:                ;; offset=0x021A
       48BF880F726B947C0000 mov      rdi, 0x7C946B720F88
       E857FB1A7D           call     CORINFO_HELP_COUNTPROFILE32
       E9AF000000           jmp      G_M000_IG18
 
G_M000_IG13:                ;; offset=0x022E
       E81D25D4FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG14:                ;; offset=0x0234
       48BF8C0F726B947C0000 mov      rdi, 0x7C946B720F8C
       E83DFB1A7D           call     CORINFO_HELP_COUNTPROFILE32
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       33C9                 xor      ecx, ecx
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       73E0                 jae      SHORT G_M000_IG13
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       48898570FFFFFF       mov      qword ptr [rbp-0x90], rax
       488B8570FFFFFF       mov      rax, qword ptr [rbp-0x90]
       488945C0             mov      qword ptr [rbp-0x40], rax
 
G_M000_IG15:                ;; offset=0x0267
       8B7DF8               mov      edi, dword ptr [rbp-0x08]
       8B75F4               mov      esi, dword ptr [rbp-0x0C]
       488B55E0             mov      rdx, qword ptr [rbp-0x20]
       488B4DC0             mov      rcx, qword ptr [rbp-0x40]
       FF152572F2FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       0FB6052FC0D5FD       movzx    rax, byte  ptr [(reloc 0x7c946925b2d1)]
       85C0                 test     eax, eax
       743C                 je       SHORT G_M000_IG17
       837DF840             cmp      dword ptr [rbp-0x08], 64
       0F8C76FFFFFF         jl       G_M000_IG11
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45C0             mov      r8, qword ptr [rbp-0x40]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15D5ED0500         call     [Lokad.Onnx.MathOps:TryPackedAvx512Rows(int,int,int,ptr,ptr,ptr):bool]
       85C0                 test     eax, eax
       0F8567FFFFFF         jne      G_M000_IG12
 
G_M000_IG16:                ;; offset=0x02B3
       48BF900F726B947C0000 mov      rdi, 0x7C946B720F90
       E8BEFA1A7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG17:                ;; offset=0x02C2
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45C0             mov      r8, qword ptr [rbp-0x40]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15BBED0500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG18:                ;; offset=0x02DD
       48BF940F726B947C0000 mov      rdi, 0x7C946B720F94
       E894FA1A7D           call     CORINFO_HELP_COUNTPROFILE32
       90                   nop      
 
G_M000_IG19:                ;; offset=0x02ED
       E857060000           call     G_M000_IG72
       EB4A                 jmp      SHORT G_M000_IG24
 
G_M000_IG20:                ;; offset=0x02F4
       48BF9C0F726B947C0000 mov      rdi, 0x7C946B720F9C
       E87DFA1A7D           call     CORINFO_HELP_COUNTPROFILE32
       EB43                 jmp      SHORT G_M000_IG25
 
G_M000_IG21:                ;; offset=0x0305
       48BFA00F726B947C0000 mov      rdi, 0x7C946B720FA0
       E86CFA1A7D           call     CORINFO_HELP_COUNTPROFILE32
       EB32                 jmp      SHORT G_M000_IG25
 
G_M000_IG22:                ;; offset=0x0316
       48BFA40F726B947C0000 mov      rdi, 0x7C946B720FA4
       E85BFA1A7D           call     CORINFO_HELP_COUNTPROFILE32
       E95A050000           jmp      G_M000_IG65
 
G_M000_IG23:                ;; offset=0x032A
       48BFA80F726B947C0000 mov      rdi, 0x7C946B720FA8
       E847FA1A7D           call     CORINFO_HELP_COUNTPROFILE32
       E946050000           jmp      G_M000_IG65
 
G_M000_IG24:                ;; offset=0x033E
       E824060000           call     G_M000_IG75
       E91B040000           jmp      G_M000_IG58
 
G_M000_IG25:                ;; offset=0x0348
       837DD440             cmp      dword ptr [rbp-0x2C], 64
       0F8CBD010000         jl       G_M000_IG40
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000004         cmp      rax, 0x4000000
       0F8F8A010000         jg       G_M000_IG38
       488D3C24             lea      rdi, [rsp]
       488D7510             lea      rsi, [rbp+0x10]
       488B0E               mov      rcx, gword ptr [rsi]
       48890C24             mov      gword ptr [rsp], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2408           mov      gword ptr [rsp+0x08], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2410           mov      gword ptr [rsp+0x10], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2418           mov      gword ptr [rsp+0x18], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2420           mov      gword ptr [rsp+0x20], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       48A5                 movsq    
       8B45F8               mov      eax, dword ptr [rbp-0x08]
       8BF8                 mov      edi, eax
       0FAF7DF4             imul     edi, dword ptr [rbp-0x0C]
       FF156EF60500         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488945B0             mov      gword ptr [rbp-0x50], rax
 
G_M000_IG26:                ;; offset=0x03D6
       488B45B0             mov      rax, gword ptr [rbp-0x50]
       488945B8             mov      gword ptr [rbp-0x48], rax
       48837DB000           cmp      gword ptr [rbp-0x50], 0
       7419                 je       SHORT G_M000_IG28
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       83780800             cmp      dword ptr [rax+0x08], 0
       7545                 jne      SHORT G_M000_IG32
 
G_M000_IG27:                ;; offset=0x03EF
       48BFB810726B947C0000 mov      rdi, 0x7C946B7210B8
       E882F91A7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG28:                ;; offset=0x03FE
       33C0                 xor      eax, eax
       488945A8             mov      qword ptr [rbp-0x58], rax
       EB61                 jmp      SHORT G_M000_IG33
 
G_M000_IG29:                ;; offset=0x0406
       48BFBC10726B947C0000 mov      rdi, 0x7C946B7210BC
       E86BF91A7D           call     CORINFO_HELP_COUNTPROFILE32
       E9A8000000           jmp      G_M000_IG35
 
G_M000_IG30:                ;; offset=0x041A
       48BFC010726B947C0000 mov      rdi, 0x7C946B7210C0
       E857F91A7D           call     CORINFO_HELP_COUNTPROFILE32
       E9AF000000           jmp      G_M000_IG36
 
G_M000_IG31:                ;; offset=0x042E
       E81D23D4FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG32:                ;; offset=0x0434
       48BFC410726B947C0000 mov      rdi, 0x7C946B7210C4
       E83DF91A7D           call     CORINFO_HELP_COUNTPROFILE32
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       33C9                 xor      ecx, ecx
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       73E0                 jae      SHORT G_M000_IG31
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       48898560FFFFFF       mov      qword ptr [rbp-0xA0], rax
       488B8560FFFFFF       mov      rax, qword ptr [rbp-0xA0]
       488945A8             mov      qword ptr [rbp-0x58], rax
 
G_M000_IG33:                ;; offset=0x0467
       8B7DF8               mov      edi, dword ptr [rbp-0x08]
       8B75F4               mov      esi, dword ptr [rbp-0x0C]
       488B55E0             mov      rdx, qword ptr [rbp-0x20]
       488B4DA8             mov      rcx, qword ptr [rbp-0x58]
       FF152570F2FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       0FB6052FBED5FD       movzx    rax, byte  ptr [(reloc 0x7c946925b2d1)]
       85C0                 test     eax, eax
       743C                 je       SHORT G_M000_IG35
       837DF840             cmp      dword ptr [rbp-0x08], 64
       0F8C76FFFFFF         jl       G_M000_IG29
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45A8             mov      r8, qword ptr [rbp-0x58]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15D5EB0500         call     [Lokad.Onnx.MathOps:TryPackedAvx512Rows(int,int,int,ptr,ptr,ptr):bool]
       85C0                 test     eax, eax
       0F8567FFFFFF         jne      G_M000_IG30
 
G_M000_IG34:                ;; offset=0x04B3
       48BFC810726B947C0000 mov      rdi, 0x7C946B7210C8
       E8BEF81A7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG35:                ;; offset=0x04C2
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45A8             mov      r8, qword ptr [rbp-0x58]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15D3EB0500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG36:                ;; offset=0x04DD
       48BFCC10726B947C0000 mov      rdi, 0x7C946B7210CC
       E894F81A7D           call     CORINFO_HELP_COUNTPROFILE32
       90                   nop      
 
G_M000_IG37:                ;; offset=0x04ED
       E8D4040000           call     G_M000_IG78
       EB11                 jmp      SHORT G_M000_IG39
 
G_M000_IG38:                ;; offset=0x04F4
       48BFD410726B947C0000 mov      rdi, 0x7C946B7210D4
       E87DF81A7D           call     CORINFO_HELP_COUNTPROFILE32
       EB0A                 jmp      SHORT G_M000_IG40
 
G_M000_IG39:                ;; offset=0x0505
       E8DA040000           call     G_M000_IG81
       E954020000           jmp      G_M000_IG58
 
G_M000_IG40:                ;; offset=0x050F
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       898544FFFFFF         mov      dword ptr [rbp-0xBC], eax
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       B903000000           mov      ecx, 3
       99                   cdq      
       F7F9                 idiv     edx:eax, ecx
       8D0440               lea      eax, [rax+2*rax]
       8B8D44FFFFFF         mov      ecx, dword ptr [rbp-0xBC]
       2BC8                 sub      ecx, eax
       85C9                 test     ecx, ecx
       741D                 je       SHORT G_M000_IG42
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       83E001               and      eax, 1
       85C0                 test     eax, eax
       0F859C010000         jne      G_M000_IG52
 
G_M000_IG41:                ;; offset=0x0540
       48BFE811726B947C0000 mov      rdi, 0x7C946B7211E8
       E831F81A7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG42:                ;; offset=0x054F
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000100         cmp      rax, 0x10000
       0F8F94010000         jg       G_M000_IG55
       488D3C24             lea      rdi, [rsp]
       488D7510             lea      rsi, [rbp+0x10]
       488B0E               mov      rcx, gword ptr [rsi]
       48890C24             mov      gword ptr [rsp], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2408           mov      gword ptr [rsp+0x08], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2410           mov      gword ptr [rsp+0x10], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2418           mov      gword ptr [rsp+0x18], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2420           mov      gword ptr [rsp+0x20], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       48A5                 movsq    
       8B45F8               mov      eax, dword ptr [rbp-0x08]
       8BF8                 mov      edi, eax
       0FAF7DF4             imul     edi, dword ptr [rbp-0x0C]
       FF1571F40500         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488945A0             mov      gword ptr [rbp-0x60], rax
 
G_M000_IG43:                ;; offset=0x05D3
       488B45A0             mov      rax, gword ptr [rbp-0x60]
       488945B8             mov      gword ptr [rbp-0x48], rax
       48837DA000           cmp      gword ptr [rbp-0x60], 0
       7419                 je       SHORT G_M000_IG45
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       83780800             cmp      dword ptr [rax+0x08], 0
       7517                 jne      SHORT G_M000_IG46
 
G_M000_IG44:                ;; offset=0x05EC
       48BFEC11726B947C0000 mov      rdi, 0x7C946B7211EC
       E885F71A7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG45:                ;; offset=0x05FB
       33C0                 xor      eax, eax
       48894598             mov      qword ptr [rbp-0x68], rax
       EB37                 jmp      SHORT G_M000_IG47
 
G_M000_IG46:                ;; offset=0x0603
       48BFF011726B947C0000 mov      rdi, 0x7C946B7211F0
       E86EF71A7D           call     CORINFO_HELP_COUNTPROFILE32
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       33C9                 xor      ecx, ecx
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       0F837C000000         jae      G_M000_IG48
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       48898550FFFFFF       mov      qword ptr [rbp-0xB0], rax
       488B8550FFFFFF       mov      rax, qword ptr [rbp-0xB0]
       48894598             mov      qword ptr [rbp-0x68], rax
 
G_M000_IG47:                ;; offset=0x063A
       8B7DF8               mov      edi, dword ptr [rbp-0x08]
       8B75F4               mov      esi, dword ptr [rbp-0x0C]
       488B55E0             mov      rdx, qword ptr [rbp-0x20]
       488B4D98             mov      rcx, qword ptr [rbp-0x68]
       FF15526EF2FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       898544FFFFFF         mov      dword ptr [rbp-0xBC], eax
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       B903000000           mov      ecx, 3
       99                   cdq      
       F7F9                 idiv     edx:eax, ecx
       8D0440               lea      eax, [rax+2*rax]
       8B8D44FFFFFF         mov      ecx, dword ptr [rbp-0xBC]
       2BC8                 sub      ecx, eax
       85C9                 test     ecx, ecx
       7532                 jne      SHORT G_M000_IG49
       48BFF411726B947C0000 mov      rdi, 0x7C946B7211F4
       E800F71A7D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B4598             mov      r8, qword ptr [rbp-0x68]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15FDE90500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       EB38                 jmp      SHORT G_M000_IG51
 
G_M000_IG48:                ;; offset=0x069D
       E8AE20D4FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG49:                ;; offset=0x06A3
       48BFF811726B947C0000 mov      rdi, 0x7C946B7211F8
       E8CEF61A7D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B4598             mov      r8, qword ptr [rbp-0x68]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15E3E90500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG50:                ;; offset=0x06CE
       E870030000           call     G_M000_IG84
       EB1F                 jmp      SHORT G_M000_IG54
 
G_M000_IG51:                ;; offset=0x06D5
       E869030000           call     G_M000_IG84
       EB11                 jmp      SHORT G_M000_IG53
 
G_M000_IG52:                ;; offset=0x06DC
       48BF0012726B947C0000 mov      rdi, 0x7C946B721200
       E895F61A7D           call     CORINFO_HELP_COUNTPROFILE32
       EB0E                 jmp      SHORT G_M000_IG55
 
G_M000_IG53:                ;; offset=0x06ED
       E86F030000           call     G_M000_IG87
       EB6F                 jmp      SHORT G_M000_IG58
 
G_M000_IG54:                ;; offset=0x06F4
       E868030000           call     G_M000_IG87
       EB68                 jmp      SHORT G_M000_IG58
 
G_M000_IG55:                ;; offset=0x06FB
       817DF8000A0000       cmp      dword ptr [rbp-0x08], 0xA00
       7D44                 jge      SHORT G_M000_IG57
       817DF4000A0000       cmp      dword ptr [rbp-0x0C], 0xA00
       7D2C                 jge      SHORT G_M000_IG56
       48BF1013726B947C0000 mov      rdi, 0x7C946B721310
       E864F61A7D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1551F30500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4tiled(int,int,int,ptr,ptr,ptr)]
       EB2A                 jmp      SHORT G_M000_IG58
 
G_M000_IG56:                ;; offset=0x0739
       48BF1413726B947C0000 mov      rdi, 0x7C946B721314
       E838F61A7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG57:                ;; offset=0x0748
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF153DF30500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG58:                ;; offset=0x0763
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       898544FFFFFF         mov      dword ptr [rbp-0xBC], eax
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       B903000000           mov      ecx, 3
       99                   cdq      
       F7F9                 idiv     edx:eax, ecx
       8D0440               lea      eax, [rax+2*rax]
       8B8D44FFFFFF         mov      ecx, dword ptr [rbp-0xBC]
       2BC8                 sub      ecx, eax
       85C9                 test     ecx, ecx
       756F                 jne      SHORT G_M000_IG61
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000004         cmp      rax, 0x4000000
       7F32                 jg       SHORT G_M000_IG59
       837DFC40             cmp      dword ptr [rbp-0x04], 64
       7D3D                 jge      SHORT G_M000_IG60
       48BF1813726B947C0000 mov      rdi, 0x7C946B721318
       E8D1F51A7D           call     CORINFO_HELP_COUNTPROFILE32
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000100         cmp      rax, 0x10000
       0F9EC0               setle    al
       0FB6C0               movzx    rax, al
       894594               mov      dword ptr [rbp-0x6C], eax
       EB2E                 jmp      SHORT G_M000_IG62
 
G_M000_IG59:                ;; offset=0x07CC
       48BF1C13726B947C0000 mov      rdi, 0x7C946B72131C
       E8A5F51A7D           call     CORINFO_HELP_COUNTPROFILE32
       EB18                 jmp      SHORT G_M000_IG61
 
G_M000_IG60:                ;; offset=0x07DD
       48BF2013726B947C0000 mov      rdi, 0x7C946B721320
       E894F51A7D           call     CORINFO_HELP_COUNTPROFILE32
       C7459401000000       mov      dword ptr [rbp-0x6C], 1
       EB05                 jmp      SHORT G_M000_IG62
 
G_M000_IG61:                ;; offset=0x07F5
       33C0                 xor      eax, eax
       894594               mov      dword ptr [rbp-0x6C], eax
 
G_M000_IG62:                ;; offset=0x07FA
       0FB64594             movzx    rax, byte  ptr [rbp-0x6C]
       8945D0               mov      dword ptr [rbp-0x30], eax
       8B45D4               mov      eax, dword ptr [rbp-0x2C]
       3B45FC               cmp      eax, dword ptr [rbp-0x04]
       7467                 je       SHORT G_M000_IG64
       837DD000             cmp      dword ptr [rbp-0x30], 0
       754D                 jne      SHORT G_M000_IG63
       48BF2413726B947C0000 mov      rdi, 0x7C946B721324
       E862F51A7D           call     CORINFO_HELP_COUNTPROFILE32
       488B45E8             mov      rax, qword ptr [rbp-0x18]
       8B4DD4               mov      ecx, dword ptr [rbp-0x2C]
       0FAF4DF8             imul     ecx, dword ptr [rbp-0x08]
       4863C9               movsxd   rcx, ecx
       488D0C88             lea      rcx, [rax+4*rcx]
       488B45D8             mov      rax, qword ptr [rbp-0x28]
       8B55D4               mov      edx, dword ptr [rbp-0x2C]
       0FAF55F4             imul     edx, dword ptr [rbp-0x0C]
       4863D2               movsxd   rdx, edx
       4C8D0C90             lea      r9, [rax+4*rdx]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       BF01000000           mov      edi, 1
       FF1561F20500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E9E3000000           jmp      G_M000_IG70
 
G_M000_IG63:                ;; offset=0x085C
       48BF2813726B947C0000 mov      rdi, 0x7C946B721328
       E815F51A7D           call     CORINFO_HELP_COUNTPROFILE32
       E9C0000000           jmp      G_M000_IG69
 
G_M000_IG64:                ;; offset=0x0870
       48BF2C13726B947C0000 mov      rdi, 0x7C946B72132C
       E801F51A7D           call     CORINFO_HELP_COUNTPROFILE32
       E9AC000000           jmp      G_M000_IG69
 
G_M000_IG65:                ;; offset=0x0884
       488D7D10             lea      rdi, [rbp+0x10]
       FF159AC7F2FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       7449                 je       SHORT G_M000_IG67
       488D7D10             lea      rdi, [rbp+0x10]
       FF1574C7F2FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseIntrinsics():bool:this]
       85C0                 test     eax, eax
       742C                 je       SHORT G_M000_IG66
       48BF3013726B947C0000 mov      rdi, 0x7C946B721330
       E8D1F41A7D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15EEF10500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       EB73                 jmp      SHORT G_M000_IG70
 
G_M000_IG66:                ;; offset=0x08CC
       48BF3413726B947C0000 mov      rdi, 0x7C946B721334
       E8A5F41A7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG67:                ;; offset=0x08DB
       488D7D10             lea      rdi, [rbp+0x10]
       FF1543C7F2FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       742C                 je       SHORT G_M000_IG68
       48BF3813726B947C0000 mov      rdi, 0x7C946B721338
       E888F41A7D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15BDF10500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized(int,int,int,ptr,ptr,ptr)]
       EB2A                 jmp      SHORT G_M000_IG70
 
G_M000_IG68:                ;; offset=0x0915
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15B8F10500         call     [Lokad.Onnx.MathOps:mm(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG69:                ;; offset=0x0930
       48BF3C13726B947C0000 mov      rdi, 0x7C946B72133C
       E841F41A7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG70:                ;; offset=0x093F
       90                   nop      
 
G_M000_IG71:                ;; offset=0x0940
       4881C4F0000000       add      rsp, 240
       5D                   pop      rbp
       C3                   ret      
 
G_M000_IG72:                ;; offset=0x0949
       4883EC38             sub      rsp, 56
 
G_M000_IG73:                ;; offset=0x094D
       48BF980F726B947C0000 mov      rdi, 0x7C946B720F98
       E824F41A7D           call     CORINFO_HELP_COUNTPROFILE32
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
 
G_M000_IG74:                ;; offset=0x0962
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG75:                ;; offset=0x0967
       4883EC38             sub      rsp, 56
 
G_M000_IG76:                ;; offset=0x096B
       48BFAC0F726B947C0000 mov      rdi, 0x7C946B720FAC
       E806F41A7D           call     CORINFO_HELP_COUNTPROFILE32
       FF1580F10500         call     [System.Buffers.ArrayPool`1[float]:get_Shared():System.Buffers.ArrayPool`1[float]]
       48894588             mov      gword ptr [rbp-0x78], rax
       488B7D88             mov      rdi, gword ptr [rbp-0x78]
       48BEB00F726B947C0000 mov      rsi, 0x7C946B720FB0
       E8F9EF1A7D           call     CORINFO_HELP_CLASSPROFILE32
       488B4588             mov      rax, gword ptr [rbp-0x78]
       48898568FFFFFF       mov      gword ptr [rbp-0x98], rax
       488BBD68FFFFFF       mov      rdi, gword ptr [rbp-0x98]
       488B75C8             mov      rsi, gword ptr [rbp-0x38]
       33D2                 xor      edx, edx
       488B8568FFFFFF       mov      rax, gword ptr [rbp-0x98]
       488B00               mov      rax, qword ptr [rax]
       488B4040             mov      rax, qword ptr [rax+0x40]
       FF5028               call     [rax+0x28]System.Buffers.ArrayPool`1[float]:Return(float[],bool):this
       90                   nop      
 
G_M000_IG77:                ;; offset=0x09C1
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG78:                ;; offset=0x09C6
       4883EC38             sub      rsp, 56
 
G_M000_IG79:                ;; offset=0x09CA
       48BFD010726B947C0000 mov      rdi, 0x7C946B7210D0
       E8A7F31A7D           call     CORINFO_HELP_COUNTPROFILE32
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
 
G_M000_IG80:                ;; offset=0x09DF
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG81:                ;; offset=0x09E4
       4883EC38             sub      rsp, 56
 
G_M000_IG82:                ;; offset=0x09E8
       48BFD810726B947C0000 mov      rdi, 0x7C946B7210D8
       E889F31A7D           call     CORINFO_HELP_COUNTPROFILE32
       FF1503F10500         call     [System.Buffers.ArrayPool`1[float]:get_Shared():System.Buffers.ArrayPool`1[float]]
       48894580             mov      gword ptr [rbp-0x80], rax
       488B7D80             mov      rdi, gword ptr [rbp-0x80]
       48BEE010726B947C0000 mov      rsi, 0x7C946B7210E0
       E87CEF1A7D           call     CORINFO_HELP_CLASSPROFILE32
       488B4580             mov      rax, gword ptr [rbp-0x80]
       48898558FFFFFF       mov      gword ptr [rbp-0xA8], rax
       488BBD58FFFFFF       mov      rdi, gword ptr [rbp-0xA8]
       488B75B0             mov      rsi, gword ptr [rbp-0x50]
       33D2                 xor      edx, edx
       488B8558FFFFFF       mov      rax, gword ptr [rbp-0xA8]
       488B00               mov      rax, qword ptr [rax]
       488B4040             mov      rax, qword ptr [rax+0x40]
       FF5028               call     [rax+0x28]System.Buffers.ArrayPool`1[float]:Return(float[],bool):this
       90                   nop      
 
G_M000_IG83:                ;; offset=0x0A3E
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG84:                ;; offset=0x0A43
       4883EC38             sub      rsp, 56
 
G_M000_IG85:                ;; offset=0x0A47
       48BFFC11726B947C0000 mov      rdi, 0x7C946B7211FC
       E82AF31A7D           call     CORINFO_HELP_COUNTPROFILE32
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
 
G_M000_IG86:                ;; offset=0x0A5C
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG87:                ;; offset=0x0A61
       4883EC38             sub      rsp, 56
 
G_M000_IG88:                ;; offset=0x0A65
       48BF0412726B947C0000 mov      rdi, 0x7C946B721204
       E80CF31A7D           call     CORINFO_HELP_COUNTPROFILE32
       FF1586F00500         call     [System.Buffers.ArrayPool`1[float]:get_Shared():System.Buffers.ArrayPool`1[float]]
       48898578FFFFFF       mov      gword ptr [rbp-0x88], rax
       488BBD78FFFFFF       mov      rdi, gword ptr [rbp-0x88]
       48BE0812726B947C0000 mov      rsi, 0x7C946B721208
       E8F9EE1A7D           call     CORINFO_HELP_CLASSPROFILE32
       488B8578FFFFFF       mov      rax, gword ptr [rbp-0x88]
       48898548FFFFFF       mov      gword ptr [rbp-0xB8], rax
       488BBD48FFFFFF       mov      rdi, gword ptr [rbp-0xB8]
       488B75A0             mov      rsi, gword ptr [rbp-0x60]
       33D2                 xor      edx, edx
       488B8548FFFFFF       mov      rax, gword ptr [rbp-0xB8]
       488B00               mov      rax, qword ptr [rax]
       488B4040             mov      rax, qword ptr [rax+0x40]
       FF5028               call     [rax+0x28]System.Buffers.ArrayPool`1[float]:Return(float[],bool):this
       90                   nop      
 
G_M000_IG89:                ;; offset=0x0AC4
       4883C438             add      rsp, 56
       C3                   ret      
 
; Total bytes of code 2761
