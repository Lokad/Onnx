; Assembly listing for method Lokad.Onnx.Tensor`1[float]:RunFloatMatMulKernel(int,int,int,ptr,ptr,ptr,Lokad.Onnx.TensorExecutionOptions) (Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       55                   push     rbp
       4881ECE0000000       sub      rsp, 224
       488DAC24E0000000     lea      rbp, [rsp+0xE0]
       C4413857C0           vxorps   xmm8, xmm8, xmm8
       62717E487F8560FFFFFF vmovdqu32 zmmword ptr [rbp-0xA0], zmm8
       62717E487F8590FFFFFF vmovdqu32 zmmword ptr [rbp-0x70], zmm8
       33C0                 xor      eax, eax
       488945D0             mov      qword ptr [rbp-0x30], rax
       897DFC               mov      dword ptr [rbp-0x04], edi
       8975F8               mov      dword ptr [rbp-0x08], esi
       8955F4               mov      dword ptr [rbp-0x0C], edx
       48894DE8             mov      qword ptr [rbp-0x18], rcx
       4C8945E0             mov      qword ptr [rbp-0x20], r8
       4C894DD8             mov      qword ptr [rbp-0x28], r9
 
G_M000_IG02:                ;; offset=0x0044
       488D7D10             lea      rdi, [rbp+0x10]
       FF15AA9BF6FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       743D                 je       SHORT G_M000_IG03
       488D7D10             lea      rdi, [rbp+0x10]
       FF15849BF6FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseIntrinsics():bool:this]
       85C0                 test     eax, eax
       742F                 je       SHORT G_M000_IG03
       837DFC01             cmp      dword ptr [rbp-0x04], 1
       7529                 jne      SHORT G_M000_IG03
       817DF400200000       cmp      dword ptr [rbp-0x0C], 0x2000
       7C20                 jl       SHORT G_M000_IG03
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF150EC80800         call     [Lokad.Onnx.MathOps:mm_m1_kblocked(int,int,int,ptr,ptr,ptr)]
       E911060000           jmp      G_M000_IG45
 
G_M000_IG03:                ;; offset=0x008F
       488D7D10             lea      rdi, [rbp+0x10]
       FF155F9BF6FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       0F8480050000         je       G_M000_IG42
       488D7D10             lea      rdi, [rbp+0x10]
       FF15359BF6FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseIntrinsics():bool:this]
       85C0                 test     eax, eax
       0F846E050000         je       G_M000_IG42
       837DFC02             cmp      dword ptr [rbp-0x04], 2
       0F8C64050000         jl       G_M000_IG42
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
       89855CFFFFFF         mov      dword ptr [rbp-0xA4], eax
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       B903000000           mov      ecx, 3
       99                   cdq      
       F7F9                 idiv     edx:eax, ecx
       8D0440               lea      eax, [rax+2*rax]
       8B8D5CFFFFFF         mov      ecx, dword ptr [rbp-0xA4]
       2BC8                 sub      ecx, eax
       85C9                 test     ecx, ecx
       0F8543010000         jne      G_M000_IG14
       837DFC40             cmp      dword ptr [rbp-0x04], 64
       0F8C39010000         jl       G_M000_IG14
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000004         cmp      rax, 0x4000000
       0F8F21010000         jg       G_M000_IG14
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
 
G_M000_IG04:                ;; offset=0x0169
       48894C2420           mov      gword ptr [rsp+0x20], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       48A5                 movsq    
       8B45F8               mov      eax, dword ptr [rbp-0x08]
       8BF8                 mov      edi, eax
       0FAF7DF4             imul     edi, dword ptr [rbp-0x0C]
       FF15E1C60800         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488945C8             mov      gword ptr [rbp-0x38], rax
 
G_M000_IG05:                ;; offset=0x018B
       488B45C8             mov      rax, gword ptr [rbp-0x38]
       488945B8             mov      gword ptr [rbp-0x48], rax
       48837DC800           cmp      gword ptr [rbp-0x38], 0
       740A                 je       SHORT G_M000_IG06
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       83780800             cmp      dword ptr [rax+0x08], 0
       750E                 jne      SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x01A4
       33C0                 xor      eax, eax
       488945C0             mov      qword ptr [rbp-0x40], rax
       EB24                 jmp      SHORT G_M000_IG09
 
G_M000_IG07:                ;; offset=0x01AC
       E86FF1D6FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG08:                ;; offset=0x01B2
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       33C9                 xor      ecx, ecx
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       73EF                 jae      SHORT G_M000_IG07
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       48894588             mov      qword ptr [rbp-0x78], rax
       488B4588             mov      rax, qword ptr [rbp-0x78]
       488945C0             mov      qword ptr [rbp-0x40], rax
 
G_M000_IG09:                ;; offset=0x01D0
       8B7DF8               mov      edi, dword ptr [rbp-0x08]
       8B75F4               mov      esi, dword ptr [rbp-0x0C]
       488B55E0             mov      rdx, qword ptr [rbp-0x20]
       488B4DC0             mov      rcx, qword ptr [rbp-0x40]
       FF158C3EF6FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       0FB605968CD8FD       movzx    rax, byte  ptr [(reloc 0x74d0fec6b2d1)]
       85C0                 test     eax, eax
       7425                 je       SHORT G_M000_IG10
       837DF840             cmp      dword ptr [rbp-0x08], 64
       7C1F                 jl       SHORT G_M000_IG10
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45C0             mov      r8, qword ptr [rbp-0x40]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1598BC0800         call     [Lokad.Onnx.MathOps:TryPackedAvx512Rows(int,int,int,ptr,ptr,ptr):bool]
       85C0                 test     eax, eax
       751B                 jne      SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x0214
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45C0             mov      r8, qword ptr [rbp-0x40]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1591BC0800         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG11:                ;; offset=0x022F
       90                   nop      
 
G_M000_IG12:                ;; offset=0x0230
       E875040000           call     G_M000_IG47
       90                   nop      
 
G_M000_IG13:                ;; offset=0x0236
       E87E040000           call     G_M000_IG50
       E91E030000           jmp      G_M000_IG38
 
G_M000_IG14:                ;; offset=0x0240
       837DD440             cmp      dword ptr [rbp-0x2C], 64
       0F8C3F010000         jl       G_M000_IG24
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000004         cmp      rax, 0x4000000
       0F8F27010000         jg       G_M000_IG24
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
       FF159EC50800         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488945B0             mov      gword ptr [rbp-0x50], rax
 
G_M000_IG15:                ;; offset=0x02CE
       488B45B0             mov      rax, gword ptr [rbp-0x50]
       488945B8             mov      gword ptr [rbp-0x48], rax
       48837DB000           cmp      gword ptr [rbp-0x50], 0
       740A                 je       SHORT G_M000_IG16
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       83780800             cmp      dword ptr [rax+0x08], 0
       750E                 jne      SHORT G_M000_IG18
 
G_M000_IG16:                ;; offset=0x02E7
       33C0                 xor      eax, eax
       488945A8             mov      qword ptr [rbp-0x58], rax
       EB2A                 jmp      SHORT G_M000_IG19
 
G_M000_IG17:                ;; offset=0x02EF
       E82CF0D6FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG18:                ;; offset=0x02F5
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       33C9                 xor      ecx, ecx
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       73EF                 jae      SHORT G_M000_IG17
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       48898578FFFFFF       mov      qword ptr [rbp-0x88], rax
       488B8578FFFFFF       mov      rax, qword ptr [rbp-0x88]
       488945A8             mov      qword ptr [rbp-0x58], rax
 
G_M000_IG19:                ;; offset=0x0319
       8B7DF8               mov      edi, dword ptr [rbp-0x08]
       8B75F4               mov      esi, dword ptr [rbp-0x0C]
       488B55E0             mov      rdx, qword ptr [rbp-0x20]
       488B4DA8             mov      rcx, qword ptr [rbp-0x58]
       FF15433DF6FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       0FB6054D8BD8FD       movzx    rax, byte  ptr [(reloc 0x74d0fec6b2d1)]
       85C0                 test     eax, eax
       7425                 je       SHORT G_M000_IG20
       837DF840             cmp      dword ptr [rbp-0x08], 64
       7C1F                 jl       SHORT G_M000_IG20
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45A8             mov      r8, qword ptr [rbp-0x58]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF154FBB0800         call     [Lokad.Onnx.MathOps:TryPackedAvx512Rows(int,int,int,ptr,ptr,ptr):bool]
       85C0                 test     eax, eax
       751B                 jne      SHORT G_M000_IG21
 
G_M000_IG20:                ;; offset=0x035D
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45A8             mov      r8, qword ptr [rbp-0x58]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1560BB0800         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG21:                ;; offset=0x0378
       90                   nop      
 
G_M000_IG22:                ;; offset=0x0379
       E867030000           call     G_M000_IG53
       90                   nop      
 
G_M000_IG23:                ;; offset=0x037F
       E870030000           call     G_M000_IG56
       E9D5010000           jmp      G_M000_IG38
 
G_M000_IG24:                ;; offset=0x0389
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       89855CFFFFFF         mov      dword ptr [rbp-0xA4], eax
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       B903000000           mov      ecx, 3
       99                   cdq      
       F7F9                 idiv     edx:eax, ecx
       8D0440               lea      eax, [rax+2*rax]
       8B8D5CFFFFFF         mov      ecx, dword ptr [rbp-0xA4]
       2BC8                 sub      ecx, eax
       85C9                 test     ecx, ecx
       740E                 je       SHORT G_M000_IG25
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       83E001               and      eax, 1
       85C0                 test     eax, eax
       0F855A010000         jne      G_M000_IG36
 
G_M000_IG25:                ;; offset=0x03BA
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000100         cmp      rax, 0x10000
       0F8F42010000         jg       G_M000_IG36
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
       FF152EC40800         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488945A0             mov      gword ptr [rbp-0x60], rax
 
G_M000_IG26:                ;; offset=0x043E
       488B45A0             mov      rax, gword ptr [rbp-0x60]
       488945B8             mov      gword ptr [rbp-0x48], rax
       48837DA000           cmp      gword ptr [rbp-0x60], 0
       740A                 je       SHORT G_M000_IG27
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       83780800             cmp      dword ptr [rax+0x08], 0
       7508                 jne      SHORT G_M000_IG28
 
G_M000_IG27:                ;; offset=0x0457
       33C0                 xor      eax, eax
       48894598             mov      qword ptr [rbp-0x68], rax
       EB24                 jmp      SHORT G_M000_IG29
 
G_M000_IG28:                ;; offset=0x045F
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       33C9                 xor      ecx, ecx
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       736D                 jae      SHORT G_M000_IG30
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       48898568FFFFFF       mov      qword ptr [rbp-0x98], rax
       488B8568FFFFFF       mov      rax, qword ptr [rbp-0x98]
       48894598             mov      qword ptr [rbp-0x68], rax
 
G_M000_IG29:                ;; offset=0x0483
       8B7DF8               mov      edi, dword ptr [rbp-0x08]
       8B75F4               mov      esi, dword ptr [rbp-0x0C]
       488B55E0             mov      rdx, qword ptr [rbp-0x20]
       488B4D98             mov      rcx, qword ptr [rbp-0x68]
       FF15D93BF6FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       89855CFFFFFF         mov      dword ptr [rbp-0xA4], eax
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       B903000000           mov      ecx, 3
       99                   cdq      
       F7F9                 idiv     edx:eax, ecx
       8D0440               lea      eax, [rax+2*rax]
       8B8D5CFFFFFF         mov      ecx, dword ptr [rbp-0xA4]
       2BC8                 sub      ecx, eax
       85C9                 test     ecx, ecx
       7523                 jne      SHORT G_M000_IG31
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B4598             mov      r8, qword ptr [rbp-0x68]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15EBB90800         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       EB29                 jmp      SHORT G_M000_IG33
 
G_M000_IG30:                ;; offset=0x04D7
       E844EED6FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG31:                ;; offset=0x04DD
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B4598             mov      r8, qword ptr [rbp-0x68]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF15E0B90800         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG32:                ;; offset=0x04F9
       E82B020000           call     G_M000_IG59
       EB0D                 jmp      SHORT G_M000_IG35
 
G_M000_IG33:                ;; offset=0x0500
       E824020000           call     G_M000_IG59
       90                   nop      
 
G_M000_IG34:                ;; offset=0x0506
       E82D020000           call     G_M000_IG62
       EB51                 jmp      SHORT G_M000_IG38
 
G_M000_IG35:                ;; offset=0x050D
       E826020000           call     G_M000_IG62
       EB4A                 jmp      SHORT G_M000_IG38
 
G_M000_IG36:                ;; offset=0x0514
       817DF8000A0000       cmp      dword ptr [rbp-0x08], 0xA00
       7D26                 jge      SHORT G_M000_IG37
       817DF4000A0000       cmp      dword ptr [rbp-0x0C], 0xA00
       7D1D                 jge      SHORT G_M000_IG37
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF156FC30800         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4tiled(int,int,int,ptr,ptr,ptr)]
       EB1B                 jmp      SHORT G_M000_IG38
 
G_M000_IG37:                ;; offset=0x0543
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF156AC30800         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG38:                ;; offset=0x055E
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       89855CFFFFFF         mov      dword ptr [rbp-0xA4], eax
       8B45FC               mov      eax, dword ptr [rbp-0x04]
       B903000000           mov      ecx, 3
       99                   cdq      
       F7F9                 idiv     edx:eax, ecx
       8D0440               lea      eax, [rax+2*rax]
       8B8D5CFFFFFF         mov      ecx, dword ptr [rbp-0xA4]
       2BC8                 sub      ecx, eax
       85C9                 test     ecx, ecx
       7540                 jne      SHORT G_M000_IG40
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000004         cmp      rax, 0x4000000
       7F2C                 jg       SHORT G_M000_IG40
       837DFC40             cmp      dword ptr [rbp-0x04], 64
       7D1D                 jge      SHORT G_M000_IG39
       486345F8             movsxd   rax, dword ptr [rbp-0x08]
       48634DF4             movsxd   rcx, dword ptr [rbp-0x0C]
       480FAFC1             imul     rax, rcx
       483D00000100         cmp      rax, 0x10000
       0F9EC0               setle    al
       0FB6C0               movzx    rax, al
       894594               mov      dword ptr [rbp-0x6C], eax
       EB0E                 jmp      SHORT G_M000_IG41
 
G_M000_IG39:                ;; offset=0x05B8
       C7459401000000       mov      dword ptr [rbp-0x6C], 1
       EB05                 jmp      SHORT G_M000_IG41
 
G_M000_IG40:                ;; offset=0x05C1
       33C0                 xor      eax, eax
       894594               mov      dword ptr [rbp-0x6C], eax
 
G_M000_IG41:                ;; offset=0x05C6
       0FB64594             movzx    rax, byte  ptr [rbp-0x6C]
       8945D0               mov      dword ptr [rbp-0x30], eax
       8B45D4               mov      eax, dword ptr [rbp-0x2C]
       3B45FC               cmp      eax, dword ptr [rbp-0x04]
       0F84C7000000         je       G_M000_IG45
       837DD000             cmp      dword ptr [rbp-0x30], 0
       0F85BD000000         jne      G_M000_IG45
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
       FF15C4C20800         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E97F000000           jmp      G_M000_IG45
 
G_M000_IG42:                ;; offset=0x0621
       488D7D10             lea      rdi, [rbp+0x10]
       FF15CD95F6FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       742B                 je       SHORT G_M000_IG43
       488D7D10             lea      rdi, [rbp+0x10]
       FF15A795F6FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseIntrinsics():bool:this]
       85C0                 test     eax, eax
       741D                 je       SHORT G_M000_IG43
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1588C20800         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       EB46                 jmp      SHORT G_M000_IG45
 
G_M000_IG43:                ;; offset=0x065A
       488D7D10             lea      rdi, [rbp+0x10]
       FF159495F6FF         call     [Lokad.Onnx.TensorExecutionOptions:get_UseSimd():bool:this]
       85C0                 test     eax, eax
       741D                 je       SHORT G_M000_IG44
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1575C20800         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized(int,int,int,ptr,ptr,ptr)]
       EB1B                 jmp      SHORT G_M000_IG45
 
G_M000_IG44:                ;; offset=0x0685
       8B7DFC               mov      edi, dword ptr [rbp-0x04]
       8B75F8               mov      esi, dword ptr [rbp-0x08]
       8B55F4               mov      edx, dword ptr [rbp-0x0C]
       488B4DE8             mov      rcx, qword ptr [rbp-0x18]
       4C8B45E0             mov      r8, qword ptr [rbp-0x20]
       4C8B4DD8             mov      r9, qword ptr [rbp-0x28]
       FF1570C20800         call     [Lokad.Onnx.MathOps:mm(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG45:                ;; offset=0x06A0
       90                   nop      
 
G_M000_IG46:                ;; offset=0x06A1
       4881C4E0000000       add      rsp, 224
       5D                   pop      rbp
       C3                   ret      
 
G_M000_IG47:                ;; offset=0x06AA
       4883EC38             sub      rsp, 56
 
G_M000_IG48:                ;; offset=0x06AE
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
 
G_M000_IG49:                ;; offset=0x06B4
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG50:                ;; offset=0x06B9
       4883EC38             sub      rsp, 56
 
G_M000_IG51:                ;; offset=0x06BD
       FF1565C20800         call     [System.Buffers.ArrayPool`1[float]:get_Shared():System.Buffers.ArrayPool`1[float]]
       48894580             mov      gword ptr [rbp-0x80], rax
       488B7D80             mov      rdi, gword ptr [rbp-0x80]
       488B75C8             mov      rsi, gword ptr [rbp-0x38]
       33D2                 xor      edx, edx
       488B4580             mov      rax, gword ptr [rbp-0x80]
       488B00               mov      rax, qword ptr [rax]
       488B4040             mov      rax, qword ptr [rax+0x40]
       FF5028               call     [rax+0x28]System.Buffers.ArrayPool`1[float]:Return(float[],bool):this
       90                   nop      
 
G_M000_IG52:                ;; offset=0x06E0
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG53:                ;; offset=0x06E5
       4883EC38             sub      rsp, 56
 
G_M000_IG54:                ;; offset=0x06E9
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
 
G_M000_IG55:                ;; offset=0x06EF
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG56:                ;; offset=0x06F4
       4883EC38             sub      rsp, 56
 
G_M000_IG57:                ;; offset=0x06F8
       FF152AC20800         call     [System.Buffers.ArrayPool`1[float]:get_Shared():System.Buffers.ArrayPool`1[float]]
       48898570FFFFFF       mov      gword ptr [rbp-0x90], rax
       488BBD70FFFFFF       mov      rdi, gword ptr [rbp-0x90]
       488B75B0             mov      rsi, gword ptr [rbp-0x50]
       33D2                 xor      edx, edx
       488B8570FFFFFF       mov      rax, gword ptr [rbp-0x90]
       488B00               mov      rax, qword ptr [rax]
       488B4040             mov      rax, qword ptr [rax+0x40]
       FF5028               call     [rax+0x28]System.Buffers.ArrayPool`1[float]:Return(float[],bool):this
       90                   nop      
 
G_M000_IG58:                ;; offset=0x0724
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG59:                ;; offset=0x0729
       4883EC38             sub      rsp, 56
 
G_M000_IG60:                ;; offset=0x072D
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
 
G_M000_IG61:                ;; offset=0x0733
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG62:                ;; offset=0x0738
       4883EC38             sub      rsp, 56
 
G_M000_IG63:                ;; offset=0x073C
       FF15E6C10800         call     [System.Buffers.ArrayPool`1[float]:get_Shared():System.Buffers.ArrayPool`1[float]]
       48898560FFFFFF       mov      gword ptr [rbp-0xA0], rax
       488BBD60FFFFFF       mov      rdi, gword ptr [rbp-0xA0]
       488B75A0             mov      rsi, gword ptr [rbp-0x60]
       33D2                 xor      edx, edx
       488B8560FFFFFF       mov      rax, gword ptr [rbp-0xA0]
       488B00               mov      rax, qword ptr [rax]
       488B4040             mov      rax, qword ptr [rax+0x40]
       FF5028               call     [rax+0x28]System.Buffers.ArrayPool`1[float]:Return(float[],bool):this
       90                   nop      
 
G_M000_IG64:                ;; offset=0x0768
       4883C438             add      rsp, 56
       C3                   ret      
 
; Total bytes of code 1901
