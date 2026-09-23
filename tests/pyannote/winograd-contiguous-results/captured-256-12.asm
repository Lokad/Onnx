; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 576
       lea      rbp, [rsp+0x240]
       xor      eax, eax
       mov      qword ptr [rbp-0x1D8], rax
       vxorps   xmm8, xmm8, xmm8
       mov      rax, -384
       vmovdqa  xmmword ptr [rbp+rax-0x50], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x50], rax
       mov      qword ptr [rbp-0x30], rdi
       mov      qword ptr [rbp-0x38], rsi
       mov      dword ptr [rbp-0x3C], edx
       mov      dword ptr [rbp-0x40], ecx
       mov      dword ptr [rbp-0x44], r8d
       mov      dword ptr [rbp-0x48], r9d
 
G_M000_IG02:                ;; offset=0x005A
       mov      dword ptr [rbp-0x240], 0x3E8
       mov      eax, dword ptr [rbp-0x40]
       imul     eax, dword ptr [rbp-0x44]
       mov      dword ptr [rbp-0x4C], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x50], eax
       jmp      G_M000_IG13
 
G_M000_IG03:                ;; offset=0x0078
       xor      eax, eax
       mov      dword ptr [rbp-0x54], eax
       jmp      G_M000_IG10
 
G_M000_IG04:                ;; offset=0x0082
       mov      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x30]
       mov      ecx, dword ptr [rbp-0x54]
       shl      ecx, 3
       movsxd   rcx, ecx
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x60], rax
       mov      eax, dword ptr [rbp-0x3C]
       shl      eax, 3
       mov      dword ptr [rbp-0x64], eax
       mov      eax, dword ptr [rbp+0x10]
       add      eax, dword ptr [rbp-0x54]
       cdq      
       idiv     edx:eax, dword ptr [rbp-0x48]
       add      eax, eax
       mov      dword ptr [rbp-0x68], eax
       mov      eax, dword ptr [rbp+0x10]
       add      eax, dword ptr [rbp-0x54]
       cdq      
       idiv     edx:eax, dword ptr [rbp-0x48]
       lea      eax, [rdx+rdx]
       mov      dword ptr [rbp-0x6C], eax
       mov      rax, qword ptr [rbp-0x60]
       vmovups  ymm0, ymmword ptr [rax]
       mov      eax, dword ptr [rbp-0x64]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x90], ymm0
       mov      eax, dword ptr [rbp-0x64]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  ymm0, ymmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+2*rax]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x210], ymm0
       movsxd   rax, dword ptr [rbp-0x64]
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  ymm0, ymmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
 
G_M000_IG05:                ;; offset=0x0154
       vaddps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0xB0], ymm0
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  ymm0, ymmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 13
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0xD0], ymm0
       mov      eax, dword ptr [rbp-0x64]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  ymm0, ymmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+2*rax]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x230], ymm0
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+4*rax]
       add      eax, eax
       mov      dword ptr [rbp-0x234], eax
       movsxd   rax, dword ptr [rbp-0x234]
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  ymm0, ymmword ptr [rbp-0x230]
       vaddps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0xF0], ymm0
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+2*rax]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  ymm0, ymmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+4*rax]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 14
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x110], ymm0
 
G_M000_IG06:                ;; offset=0x0241
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  ymm0, ymmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 7
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 11
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x130], ymm0
       imul     eax, dword ptr [rbp-0x64], 7
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  ymm0, ymmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 11
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 15
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   ymm0, ymm0, ymmword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x150], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x90]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0xB0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0xF0]
       vmovups  ymmword ptr [rbp-0x170], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0xB0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0xF0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x130]
       vmovups  ymmword ptr [rbp-0x190], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x210]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0xD0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x110]
       vmovups  ymmword ptr [rbp-0x1B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0xD0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x110]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x150]
       vmovups  ymmword ptr [rbp-0x1D0], ymm0
       mov      eax, dword ptr [rbp-0x50]
       imul     eax, dword ptr [rbp-0x4C]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       mov      ecx, dword ptr [rbp-0x68]
       imul     ecx, dword ptr [rbp-0x44]
       add      ecx, dword ptr [rbp-0x6C]
       shl      ecx, 3
       movsxd   rcx, ecx
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x1D8], rax
       mov      rax, qword ptr [rbp-0x1D8]
       vmovups  ymm0, ymmword ptr [rbp-0x170]
       vmovups  ymmword ptr [rax], ymm0
       mov      eax, dword ptr [rbp-0x6C]
       inc      eax
 
G_M000_IG07:                ;; offset=0x0371
       cmp      eax, dword ptr [rbp-0x44]
       jge      SHORT G_M000_IG08
       mov      rdi, 0x7C0BEB524B60
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x1D8]
       vmovups  ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rax+0x20], ymm0
 
G_M000_IG08:                ;; offset=0x0399
       mov      eax, dword ptr [rbp-0x68]
       inc      eax
       cmp      eax, dword ptr [rbp-0x40]
       jge      SHORT G_M000_IG09
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x1D8]
       vmovups  ymm0, ymmword ptr [rbp-0x1B0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp-0x6C]
       inc      eax
       cmp      eax, dword ptr [rbp-0x44]
       jge      G_M000_IG17
       mov      rdi, 0x7C0BEB524B64
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
       lea      eax, [8*rax+0x08]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1D8]
       vmovups  ymm0, ymmword ptr [rbp-0x1D0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG09:                ;; offset=0x03FC
       mov      rdi, 0x7C0BEB524B68
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x54]
       inc      eax
       mov      dword ptr [rbp-0x54], eax
 
G_M000_IG10:                ;; offset=0x0413
       mov      eax, dword ptr [rbp-0x240]
       dec      eax
       mov      dword ptr [rbp-0x240], eax
       cmp      dword ptr [rbp-0x240], 0
       jg       SHORT G_M000_IG12
 
G_M000_IG11:                ;; offset=0x042A
       lea      rdi, [rbp-0x240]
       mov      esi, 672
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG12:                ;; offset=0x043B
       mov      eax, dword ptr [rbp-0x54]
       cmp      eax, dword ptr [rbp+0x18]
       jl       G_M000_IG04
       mov      rdi, 0x7C0BEB524B6C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 8
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG13:                ;; offset=0x045F
       mov      eax, dword ptr [rbp-0x240]
       dec      eax
       mov      dword ptr [rbp-0x240], eax
       cmp      dword ptr [rbp-0x240], 0
       jg       SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x0476
       lea      rdi, [rbp-0x240]
       mov      esi, 684
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG15:                ;; offset=0x0487
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp-0x3C]
       jl       G_M000_IG03
       mov      rdi, 0x7C0BEB524B70
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG16:                ;; offset=0x04A3
       vzeroupper 
       add      rsp, 576
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x04AF
       mov      rdi, 0x7C0BEB524B74
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG09
 
; Total bytes of code 1219

