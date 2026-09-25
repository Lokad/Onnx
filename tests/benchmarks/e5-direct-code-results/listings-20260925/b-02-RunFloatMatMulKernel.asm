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
       FF15A3ADF3FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       0F847D000000         je       G_M000_IG06
       488D7D10             lea      rdi, [rbp+0x10]
       FF1579ADF3FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseIntrinsics():bool:this]
       85C0                 test     eax, eax
       7460                 je       SHORT G_M000_IG05
       837DFC01             cmp      dword ptr [rbp-0x04], 1
       7549                 jne      SHORT G_M000_IG04
       817DF400200000       cmp      dword ptr [rbp-0x0C], 0x2000
       7C2F                 jl       SHORT G_M000_IG03
       48BF88661201D1740000 mov      rdi, 0x74D101126688
       E8C7DA197D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15F4D90500         call     [Lokad.Onnx.MathOps:mm_m1_kblocked(int,int,int,ptr,ptr,ptr)]
       E996080000           jmp      G_M000_IG70
 
G_M000_IG03:                ;; offset=0x00A9
       48BF8C661201D1740000 mov      rdi, 0x74D10112668C
       E898DA197D           call     CORINFO_HELP_COUNTPROFILE32
       EB20                 jmp      SHORT G_M000_IG06
 
G_M000_IG04:                ;; offset=0x00BA
       48BF90661201D1740000 mov      rdi, 0x74D101126690
       E887DA197D           call     CORINFO_HELP_COUNTPROFILE32
       EB0F                 jmp      SHORT G_M000_IG06
 
G_M000_IG05:                ;; offset=0x00CB
       48BF94661201D1740000 mov      rdi, 0x74D101126694
       E876DA197D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG06:                ;; offset=0x00DA
       488D7D10             lea      rdi, [rbp+0x10]
       FF1514ADF3FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       0F8498070000         je       G_M000_IG65
       488D7D10             lea      rdi, [rbp+0x10]
       FF15EAACF3FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseIntrinsics():bool:this]
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
       FF1596D80500         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
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
       48BF98661201D1740000 mov      rdi, 0x74D101126698
       E852D9197D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG10:                ;; offset=0x01FE
       33C0                 xor      eax, eax
       488945C0             mov      qword ptr [rbp-0x40], rax
       EB61                 jmp      SHORT G_M000_IG15
 
G_M000_IG11:                ;; offset=0x0206
       48BF9C661201D1740000 mov      rdi, 0x74D10112669C
       E83BD9197D           call     CORINFO_HELP_COUNTPROFILE32
       E9A8000000           jmp      G_M000_IG17
 
G_M000_IG12:                ;; offset=0x021A
       48BFA0661201D1740000 mov      rdi, 0x74D1011266A0
       E827D9197D           call     CORINFO_HELP_COUNTPROFILE32
       E9AF000000           jmp      G_M000_IG18
 
G_M000_IG13:                ;; offset=0x022E
       E8ED02D4FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG14:                ;; offset=0x0234
       48BFA4661201D1740000 mov      rdi, 0x74D1011266A4
       E80DD9197D           call     CORINFO_HELP_COUNTPROFILE32
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
       FF15F54FF3FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       0FB605FF9DD5FD       movzx    rax, byte  ptr [(reloc 0x74d0fec6b2d1)]
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
       FF15FDCD0500         call     [Lokad.Onnx.MathOps:TryPackedAvx512Rows(int,int,int,ptr,ptr,ptr):bool]
       85C0                 test     eax, eax
       0F8567FFFFFF         jne      G_M000_IG12
 
G_M000_IG16:                ;; offset=0x02B3
       48BFA8661201D1740000 mov      rdi, 0x74D1011266A8
       E88ED8197D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG17:                ;; offset=0x02C2
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45C0             mov      r8, qword ptr [rbp-0x40]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15E3CD0500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG18:                ;; offset=0x02DD
       48BFAC661201D1740000 mov      rdi, 0x74D1011266AC
       E864D8197D           call     CORINFO_HELP_COUNTPROFILE32
       90                   nop      
 
G_M000_IG19:                ;; offset=0x02ED
       E857060000           call     G_M000_IG72
       EB4A                 jmp      SHORT G_M000_IG24
 
G_M000_IG20:                ;; offset=0x02F4
       48BFB4661201D1740000 mov      rdi, 0x74D1011266B4
       E84DD8197D           call     CORINFO_HELP_COUNTPROFILE32
       EB43                 jmp      SHORT G_M000_IG25
 
G_M000_IG21:                ;; offset=0x0305
       48BFB8661201D1740000 mov      rdi, 0x74D1011266B8
       E83CD8197D           call     CORINFO_HELP_COUNTPROFILE32
       EB32                 jmp      SHORT G_M000_IG25
 
G_M000_IG22:                ;; offset=0x0316
       48BFBC661201D1740000 mov      rdi, 0x74D1011266BC
       E82BD8197D           call     CORINFO_HELP_COUNTPROFILE32
       E95A050000           jmp      G_M000_IG65
 
G_M000_IG23:                ;; offset=0x032A
       48BFC0661201D1740000 mov      rdi, 0x74D1011266C0
       E817D8197D           call     CORINFO_HELP_COUNTPROFILE32
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
       FF1596D60500         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
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
       48BFD0671201D1740000 mov      rdi, 0x74D1011267D0
       E852D7197D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG28:                ;; offset=0x03FE
       33C0                 xor      eax, eax
       488945A8             mov      qword ptr [rbp-0x58], rax
       EB61                 jmp      SHORT G_M000_IG33
 
G_M000_IG29:                ;; offset=0x0406
       48BFD4671201D1740000 mov      rdi, 0x74D1011267D4
       E83BD7197D           call     CORINFO_HELP_COUNTPROFILE32
       E9A8000000           jmp      G_M000_IG35
 
G_M000_IG30:                ;; offset=0x041A
       48BFD8671201D1740000 mov      rdi, 0x74D1011267D8
       E827D7197D           call     CORINFO_HELP_COUNTPROFILE32
       E9AF000000           jmp      G_M000_IG36
 
G_M000_IG31:                ;; offset=0x042E
       E8ED00D4FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG32:                ;; offset=0x0434
       48BFDC671201D1740000 mov      rdi, 0x74D1011267DC
       E80DD7197D           call     CORINFO_HELP_COUNTPROFILE32
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
       FF15F54DF3FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       0FB605FF9BD5FD       movzx    rax, byte  ptr [(reloc 0x74d0fec6b2d1)]
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
       FF15FDCB0500         call     [Lokad.Onnx.MathOps:TryPackedAvx512Rows(int,int,int,ptr,ptr,ptr):bool]
       85C0                 test     eax, eax
       0F8567FFFFFF         jne      G_M000_IG30
 
G_M000_IG34:                ;; offset=0x04B3
       48BFE0671201D1740000 mov      rdi, 0x74D1011267E0
       E88ED6197D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG35:                ;; offset=0x04C2
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45A8             mov      r8, qword ptr [rbp-0x58]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15FBCB0500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG36:                ;; offset=0x04DD
       48BFE4671201D1740000 mov      rdi, 0x74D1011267E4
       E864D6197D           call     CORINFO_HELP_COUNTPROFILE32
       90                   nop      
 
G_M000_IG37:                ;; offset=0x04ED
       E8D4040000           call     G_M000_IG78
       EB11                 jmp      SHORT G_M000_IG39
 
G_M000_IG38:                ;; offset=0x04F4
       48BFEC671201D1740000 mov      rdi, 0x74D1011267EC
       E84DD6197D           call     CORINFO_HELP_COUNTPROFILE32
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
       48BF00691201D1740000 mov      rdi, 0x74D101126900
       E801D6197D           call     CORINFO_HELP_COUNTPROFILE32
 
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
       FF1599D40500         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
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
       48BF04691201D1740000 mov      rdi, 0x74D101126904
       E855D5197D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG45:                ;; offset=0x05FB
       33C0                 xor      eax, eax
       48894598             mov      qword ptr [rbp-0x68], rax
       EB37                 jmp      SHORT G_M000_IG47
 
G_M000_IG46:                ;; offset=0x0603
       48BF08691201D1740000 mov      rdi, 0x74D101126908
       E83ED5197D           call     CORINFO_HELP_COUNTPROFILE32
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
       FF15224CF3FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
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
       48BF0C691201D1740000 mov      rdi, 0x74D10112690C
       E8D0D4197D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B4598             mov      r8, qword ptr [rbp-0x68]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1525CA0500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       EB38                 jmp      SHORT G_M000_IG51
 
G_M000_IG48:                ;; offset=0x069D
       E87EFED3FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG49:                ;; offset=0x06A3
       48BF10691201D1740000 mov      rdi, 0x74D101126910
       E89ED4197D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B4598             mov      r8, qword ptr [rbp-0x68]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF150BCA0500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG50:                ;; offset=0x06CE
       E870030000           call     G_M000_IG84
       EB1F                 jmp      SHORT G_M000_IG54
 
G_M000_IG51:                ;; offset=0x06D5
       E869030000           call     G_M000_IG84
       EB11                 jmp      SHORT G_M000_IG53
 
G_M000_IG52:                ;; offset=0x06DC
       48BF18691201D1740000 mov      rdi, 0x74D101126918
       E865D4197D           call     CORINFO_HELP_COUNTPROFILE32
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
       48BF286A1201D1740000 mov      rdi, 0x74D101126A28
       E834D4197D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1579D30500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4tiled(int,int,int,ptr,ptr,ptr)]
       EB2A                 jmp      SHORT G_M000_IG58
 
G_M000_IG56:                ;; offset=0x0739
       48BF2C6A1201D1740000 mov      rdi, 0x74D101126A2C
       E808D4197D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG57:                ;; offset=0x0748
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1565D30500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4(int,int,int,ptr,ptr,ptr)]
 
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
       48BF306A1201D1740000 mov      rdi, 0x74D101126A30
       E8A1D3197D           call     CORINFO_HELP_COUNTPROFILE32
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000100         cmp      rax, 0x10000
       0F9EC0               setle    al
       0FB6C0               movzx    rax, al
       894594               mov      dword ptr [rbp-0x6C], eax
       EB2E                 jmp      SHORT G_M000_IG62
 
G_M000_IG59:                ;; offset=0x07CC
       48BF346A1201D1740000 mov      rdi, 0x74D101126A34
       E875D3197D           call     CORINFO_HELP_COUNTPROFILE32
       EB18                 jmp      SHORT G_M000_IG61
 
G_M000_IG60:                ;; offset=0x07DD
       48BF386A1201D1740000 mov      rdi, 0x74D101126A38
       E864D3197D           call     CORINFO_HELP_COUNTPROFILE32
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
       48BF3C6A1201D1740000 mov      rdi, 0x74D101126A3C
       E832D3197D           call     CORINFO_HELP_COUNTPROFILE32
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
       FF1589D20500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E9E3000000           jmp      G_M000_IG70
 
G_M000_IG63:                ;; offset=0x085C
       48BF406A1201D1740000 mov      rdi, 0x74D101126A40
       E8E5D2197D           call     CORINFO_HELP_COUNTPROFILE32
       E9C0000000           jmp      G_M000_IG69
 
G_M000_IG64:                ;; offset=0x0870
       48BF446A1201D1740000 mov      rdi, 0x74D101126A44
       E8D1D2197D           call     CORINFO_HELP_COUNTPROFILE32
       E9AC000000           jmp      G_M000_IG69
 
G_M000_IG65:                ;; offset=0x0884
       488D7D10             lea      rdi, [rbp+0x10]
       FF156AA5F3FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       7449                 je       SHORT G_M000_IG67
       488D7D10             lea      rdi, [rbp+0x10]
       FF1544A5F3FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseIntrinsics():bool:this]
       85C0                 test     eax, eax
       742C                 je       SHORT G_M000_IG66
       48BF486A1201D1740000 mov      rdi, 0x74D101126A48
       E8A1D2197D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1516D20500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       EB73                 jmp      SHORT G_M000_IG70
 
G_M000_IG66:                ;; offset=0x08CC
       48BF4C6A1201D1740000 mov      rdi, 0x74D101126A4C
       E875D2197D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG67:                ;; offset=0x08DB
       488D7D10             lea      rdi, [rbp+0x10]
       FF1513A5F3FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       742C                 je       SHORT G_M000_IG68
       48BF506A1201D1740000 mov      rdi, 0x74D101126A50
       E858D2197D           call     CORINFO_HELP_COUNTPROFILE32
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15E5D10500         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized(int,int,int,ptr,ptr,ptr)]
       EB2A                 jmp      SHORT G_M000_IG70
 
G_M000_IG68:                ;; offset=0x0915
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15E0D10500         call     [Lokad.Onnx.MathOps:mm(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG69:                ;; offset=0x0930
       48BF546A1201D1740000 mov      rdi, 0x74D101126A54
       E811D2197D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG70:                ;; offset=0x093F
       90                   nop      
 
G_M000_IG71:                ;; offset=0x0940
       4881C4F0000000       add      rsp, 240
       5D                   pop      rbp
       C3                   ret      
 
G_M000_IG72:                ;; offset=0x0949
       4883EC38             sub      rsp, 56
 
G_M000_IG73:                ;; offset=0x094D
       48BFB0661201D1740000 mov      rdi, 0x74D1011266B0
       E8F4D1197D           call     CORINFO_HELP_COUNTPROFILE32
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
 
G_M000_IG74:                ;; offset=0x0962
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG75:                ;; offset=0x0967
       4883EC38             sub      rsp, 56
 
G_M000_IG76:                ;; offset=0x096B
       48BFC4661201D1740000 mov      rdi, 0x74D1011266C4
       E8D6D1197D           call     CORINFO_HELP_COUNTPROFILE32
       FF15A8D10500         call     [System.Buffers.ArrayPool`1[float]:get_Shared():System.Buffers.ArrayPool`1[float]]
       48894588             mov      gword ptr [rbp-0x78], rax
       488B7D88             mov      rdi, gword ptr [rbp-0x78]
       48BEC8661201D1740000 mov      rsi, 0x74D1011266C8
       E8C9CD197D           call     CORINFO_HELP_CLASSPROFILE32
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
       48BFE8671201D1740000 mov      rdi, 0x74D1011267E8
       E877D1197D           call     CORINFO_HELP_COUNTPROFILE32
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
 
G_M000_IG80:                ;; offset=0x09DF
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG81:                ;; offset=0x09E4
       4883EC38             sub      rsp, 56
 
G_M000_IG82:                ;; offset=0x09E8
       48BFF0671201D1740000 mov      rdi, 0x74D1011267F0
       E859D1197D           call     CORINFO_HELP_COUNTPROFILE32
       FF152BD10500         call     [System.Buffers.ArrayPool`1[float]:get_Shared():System.Buffers.ArrayPool`1[float]]
       48894580             mov      gword ptr [rbp-0x80], rax
       488B7D80             mov      rdi, gword ptr [rbp-0x80]
       48BEF8671201D1740000 mov      rsi, 0x74D1011267F8
       E84CCD197D           call     CORINFO_HELP_CLASSPROFILE32
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
       48BF14691201D1740000 mov      rdi, 0x74D101126914
       E8FAD0197D           call     CORINFO_HELP_COUNTPROFILE32
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
 
G_M000_IG86:                ;; offset=0x0A5C
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG87:                ;; offset=0x0A61
       4883EC38             sub      rsp, 56
 
G_M000_IG88:                ;; offset=0x0A65
       48BF1C691201D1740000 mov      rdi, 0x74D10112691C
       E8DCD0197D           call     CORINFO_HELP_COUNTPROFILE32
       FF15AED00500         call     [System.Buffers.ArrayPool`1[float]:get_Shared():System.Buffers.ArrayPool`1[float]]
       48898578FFFFFF       mov      gword ptr [rbp-0x88], rax
       488BBD78FFFFFF       mov      rdi, gword ptr [rbp-0x88]
       48BE20691201D1740000 mov      rsi, 0x74D101126920
       E8C9CC197D           call     CORINFO_HELP_CLASSPROFILE32
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
