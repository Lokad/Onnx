; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:OutputWinograd512(ptr,ptr,int,int,int,int,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x400
       lea      rbp, [rsp+0x400]
       xor      eax, eax
       mov      qword ptr [rbp-0x338], rax
       vxorps   xmm8, xmm8, xmm8
       vmovdqa32 xmmword ptr [rbp-0x330], xmm8
       mov      rax, -720
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
 
G_M000_IG02:                ;; offset=0x0061
       mov      dword ptr [rbp-0x400], 0x3E8
       mov      eax, dword ptr [rbp-0x40]
       imul     eax, dword ptr [rbp-0x44]
       mov      dword ptr [rbp-0x4C], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x50], eax
       jmp      G_M000_IG13
 
G_M000_IG03:                ;; offset=0x007F
       xor      eax, eax
       mov      dword ptr [rbp-0x54], eax
       jmp      G_M000_IG10
 
G_M000_IG04:                ;; offset=0x0089
       mov      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x30]
       mov      ecx, dword ptr [rbp-0x54]
       shl      ecx, 4
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
       vmovups  zmm0, zmmword ptr [rax]
       mov      eax, dword ptr [rbp-0x64]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0xB0], zmm0
       mov      eax, dword ptr [rbp-0x64]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  zmm0, zmmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+2*rax]
       shl      eax, 2
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x3B0], zmm0
       movsxd   rax, dword ptr [rbp-0x64]
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  zmm0, zmmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
 
G_M000_IG05:                ;; offset=0x016D
       vaddps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0xF0], zmm0
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+4*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  zmm0, zmmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 13
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x130], zmm0
       mov      eax, dword ptr [rbp-0x64]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  zmm0, zmmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+2*rax]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x3F0], zmm0
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+4*rax]
       add      eax, eax
       mov      dword ptr [rbp-0x3F4], eax
       movsxd   rax, dword ptr [rbp-0x3F4]
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  zmm0, zmmword ptr [rbp-0x3F0]
       vaddps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x170], zmm0
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+2*rax]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  zmm0, zmmword ptr [rcx+4*rax]
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+4*rax]
       add      eax, eax
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 14
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1B0], zmm0
 
G_M000_IG06:                ;; offset=0x027C
       mov      eax, dword ptr [rbp-0x64]
       lea      eax, [rax+2*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  zmm0, zmmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 7
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 11
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vaddps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x1F0], zmm0
       imul     eax, dword ptr [rbp-0x64], 7
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vmovups  zmm0, zmmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 11
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       imul     eax, dword ptr [rbp-0x64], 15
       cdqe     
       mov      rcx, qword ptr [rbp-0x60]
       vsubps   zmm0, zmm0, zmmword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x230], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0xB0]
       vaddps   zmm0, zmm0, zmmword ptr [rbp-0xF0]
       vaddps   zmm0, zmm0, zmmword ptr [rbp-0x170]
       vmovups  zmmword ptr [rbp-0x270], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0xF0]
       vsubps   zmm0, zmm0, zmmword ptr [rbp-0x170]
       vsubps   zmm0, zmm0, zmmword ptr [rbp-0x1F0]
       vmovups  zmmword ptr [rbp-0x2B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x3B0]
       vaddps   zmm0, zmm0, zmmword ptr [rbp-0x130]
       vaddps   zmm0, zmm0, zmmword ptr [rbp-0x1B0]
       vmovups  zmmword ptr [rbp-0x2F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x130]
       vsubps   zmm0, zmm0, zmmword ptr [rbp-0x1B0]
       vsubps   zmm0, zmm0, zmmword ptr [rbp-0x230]
       vmovups  zmmword ptr [rbp-0x330], zmm0
       mov      eax, dword ptr [rbp-0x50]
       imul     eax, dword ptr [rbp-0x4C]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       mov      ecx, dword ptr [rbp-0x68]
       imul     ecx, dword ptr [rbp-0x44]
       add      ecx, dword ptr [rbp-0x6C]
       shl      ecx, 4
       movsxd   rcx, ecx
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x338], rax
       mov      rax, qword ptr [rbp-0x338]
       vmovups  zmm0, zmmword ptr [rbp-0x270]
       vmovups  zmmword ptr [rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       inc      eax
 
G_M000_IG07:                ;; offset=0x03E0
       cmp      eax, dword ptr [rbp-0x44]
       jge      SHORT G_M000_IG08
       mov      rdi, 0x781E6C70B0D0
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x338]
       vmovups  zmm0, zmmword ptr [rbp-0x2B0]
       vmovups  zmmword ptr [rax+0x40], zmm0
 
G_M000_IG08:                ;; offset=0x040C
       mov      eax, dword ptr [rbp-0x68]
       inc      eax
       cmp      eax, dword ptr [rbp-0x40]
       jge      SHORT G_M000_IG09
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x338]
       vmovups  zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x6C]
       inc      eax
       cmp      eax, dword ptr [rbp-0x44]
       jge      G_M000_IG17
       mov      rdi, 0x781E6C70B0D4
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
       inc      eax
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x338]
       vmovups  zmm0, zmmword ptr [rbp-0x330]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG09:                ;; offset=0x0475
       mov      rdi, 0x781E6C70B0D8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x54]
       inc      eax
       mov      dword ptr [rbp-0x54], eax
 
G_M000_IG10:                ;; offset=0x048C
       mov      eax, dword ptr [rbp-0x400]
       dec      eax
       mov      dword ptr [rbp-0x400], eax
       cmp      dword ptr [rbp-0x400], 0
       jg       SHORT G_M000_IG12
 
G_M000_IG11:                ;; offset=0x04A3
       lea      rdi, [rbp-0x400]
       mov      esi, 677
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG12:                ;; offset=0x04B4
       mov      eax, dword ptr [rbp-0x54]
       cmp      eax, dword ptr [rbp+0x18]
       jl       G_M000_IG04
       mov      rdi, 0x781E6C70B0DC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 16
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG13:                ;; offset=0x04D8
       mov      eax, dword ptr [rbp-0x400]
       dec      eax
       mov      dword ptr [rbp-0x400], eax
       cmp      dword ptr [rbp-0x400], 0
       jg       SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x04EF
       lea      rdi, [rbp-0x400]
       mov      esi, 690
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG15:                ;; offset=0x0500
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp-0x3C]
       jl       G_M000_IG03
       mov      rdi, 0x781E6C70B0E0
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG16:                ;; offset=0x051C
       vzeroupper 
       add      rsp, 0x400
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x0528
       mov      rdi, 0x781E6C70B0E4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG09
 
; Total bytes of code 1340

