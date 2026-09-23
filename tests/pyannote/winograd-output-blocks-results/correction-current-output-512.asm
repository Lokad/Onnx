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
       mov      rdi, 0x7C6F79151948
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
       mov      rdi, 0x7C6F7915194C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
       inc      eax
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x338]
       vmovups  zmm0, zmmword ptr [rbp-0x330]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG09:                ;; offset=0x0475
       mov      rdi, 0x7C6F79151950
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
       mov      rdi, 0x7C6F79151954
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
       mov      rdi, 0x7C6F79151958
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG16:                ;; offset=0x051C
       vzeroupper 
       add      rsp, 0x400
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x0528
       mov      rdi, 0x7C6F7915195C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG09
 
; Total bytes of code 1340

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
       mov      rdi, 0x7C6F79151948
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
       mov      rdi, 0x7C6F7915194C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
       inc      eax
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x338]
       vmovups  zmm0, zmmword ptr [rbp-0x330]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG09:                ;; offset=0x0475
       mov      rdi, 0x7C6F79151950
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
       mov      rdi, 0x7C6F79151954
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
       mov      rdi, 0x7C6F79151958
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG16:                ;; offset=0x051C
       vzeroupper 
       add      rsp, 0x400
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x0528
       mov      rdi, 0x7C6F7915195C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG09
 
; Total bytes of code 1340

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:OutputWinograd512(ptr,ptr,int,int,int,int,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 34368

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 24
       lea      rbp, [rsp+0x40]
       mov      r10d, edx
       mov      ebx, dword ptr [rbp+0x10]
       mov      r11d, dword ptr [rbp+0x18]
 
G_M000_IG02:                ;; offset=0x001D
       mov      r15d, ecx
       imul     r15d, r8d
       mov      dword ptr [rbp-0x34], r15d
       xor      r14d, r14d
       cmp      r14d, r10d
       jl       SHORT G_M000_IG06
 
G_M000_IG03:                ;; offset=0x0030
       vzeroupper 
       add      rsp, 24
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG04:                ;; offset=0x0042
       mov      qword ptr [rbp-0x30], rdi
       mov      dword ptr [rbp+0x10], ebx
       mov      r15d, dword ptr [rbp-0x34]
 
G_M000_IG05:                ;; offset=0x004D
       add      r14d, 16
       cmp      r14d, r10d
       mov      ebx, dword ptr [rbp+0x10]
       mov      rdi, qword ptr [rbp-0x30]
       jge      SHORT G_M000_IG03
 
G_M000_IG06:                ;; offset=0x005D
       xor      r13d, r13d
       cmp      r13d, r11d
       jge      SHORT G_M000_IG04
       align    [0 bytes for IG07]
 
G_M000_IG07:                ;; offset=0x0065
       lea      eax, [8*r14]
       cdqe     
       shl      rax, 2
       mov      qword ptr [rbp-0x30], rdi
       add      rax, rdi
       mov      edx, r13d
       shl      edx, 4
       movsxd   rdx, edx
       lea      r12, [rax+4*rdx]
       lea      edx, [8*r10]
       mov      dword ptr [rbp-0x38], edx
       mov      dword ptr [rbp+0x10], ebx
       lea      eax, [rbx+r13]
       mov      dword ptr [rbp-0x40], eax
       cdq      
       idiv     edx:eax, r9d
       lea      edx, [rax+rax]
       mov      dword ptr [rbp-0x3C], edx
       mov      eax, dword ptr [rbp-0x40]
       cdq      
       idiv     edx:eax, r9d
       add      edx, edx
       vmovups  zmm0, zmmword ptr [r12]
       mov      eax, dword ptr [rbp-0x38]
       lea      ebx, [4*rax]
       movsxd   rbx, ebx
       vmovups  zmm1, zmmword ptr [r12+4*rbx]
       vaddps   zmm0, zmm0, zmm1
       lea      ebx, [8*rax]
       movsxd   rbx, ebx
       vmovups  zmm2, zmmword ptr [r12+4*rbx]
       vaddps   zmm0, zmm0, zmm2
       vsubps   zmm1, zmm1, zmm2
       lea      ebx, [rax+2*rax]
       lea      r15d, [4*rbx]
       movsxd   r15, r15d
       vsubps   zmm1, zmm1, zmmword ptr [r12+4*r15]
       movsxd   r15, eax
       vmovups  zmm2, zmmword ptr [r12+4*r15]
       lea      r15d, [rax+4*rax]
       movsxd   rdi, r15d
       vmovups  zmm3, zmmword ptr [r12+4*rdi]
       vaddps   zmm2, zmm2, zmm3
       lea      edi, [rax+8*rax]
       movsxd   rdi, edi
       vmovups  zmm4, zmmword ptr [r12+4*rdi]
       vaddps   zmm2, zmm2, zmm4
       vsubps   zmm3, zmm3, zmm4
       imul     edi, eax, 13
       movsxd   rdi, edi
       vsubps   zmm3, zmm3, zmmword ptr [r12+4*rdi]
       lea      edi, [rax+rax]
       movsxd   rdi, edi
       vmovups  zmm4, zmmword ptr [r12+4*rdi]
       lea      edi, [rbx+rbx]
       movsxd   rdi, edi
       vmovups  zmm5, zmmword ptr [r12+4*rdi]
       vaddps   zmm4, zmm4, zmm5
       add      r15d, r15d
       movsxd   rdi, r15d
       vmovups  zmm6, zmmword ptr [r12+4*rdi]
       vaddps   zmm4, zmm6, zmm4
       vsubps   zmm5, zmm5, zmm6
       imul     edi, eax, 14
       movsxd   rdi, edi
       vsubps   zmm5, zmm5, zmmword ptr [r12+4*rdi]
 
G_M000_IG08:                ;; offset=0x018C
       movsxd   rdi, ebx
       vmovups  zmm6, zmmword ptr [r12+4*rdi]
       lea      edi, [8*rax]
       sub      edi, eax
       movsxd   rdi, edi
       vmovups  zmm7, zmmword ptr [r12+4*rdi]
       vaddps   zmm6, zmm6, zmm7
       imul     edi, eax, 11
       movsxd   rdi, edi
       vmovups  zmm8, zmmword ptr [r12+4*rdi]
       vaddps   zmm6, zmm6, zmm8
       vsubps   zmm7, zmm7, zmm8
       mov      edi, eax
       shl      edi, 4
       sub      edi, eax
       movsxd   rax, edi
       vsubps   zmm7, zmm7, zmmword ptr [r12+4*rax]
       vaddps   zmm0, zmm2, zmm0
       vaddps   zmm0, zmm0, zmm4
       vsubps   zmm2, zmm2, zmm4
       vsubps   zmm2, zmm2, zmm6
       vaddps   zmm1, zmm1, zmm3
       vaddps   zmm1, zmm1, zmm5
       vsubps   zmm3, zmm3, zmm5
       vsubps   zmm3, zmm3, zmm7
       mov      edi, dword ptr [rbp-0x3C]
       mov      eax, edi
       imul     eax, r8d
       add      eax, edx
       shl      eax, 4
       cdqe     
       mov      r15d, dword ptr [rbp-0x34]
       mov      ebx, r14d
       imul     ebx, r15d
       movsxd   rbx, ebx
       shl      rbx, 2
       add      rbx, rsi
       lea      rax, [rbx+4*rax]
       vmovups  zmmword ptr [rax], zmm0
       inc      edx
       cmp      edx, r8d
       jge      SHORT G_M000_IG10
 
G_M000_IG09:                ;; offset=0x023F
       vmovups  zmmword ptr [rax+0x40], zmm2
 
G_M000_IG10:                ;; offset=0x0246
       inc      edi
       cmp      edi, ecx
       jge      SHORT G_M000_IG13
 
G_M000_IG11:                ;; offset=0x024C
       mov      edi, r8d
       shl      edi, 4
       movsxd   rdi, edi
       vmovups  zmmword ptr [rax+4*rdi], zmm1
       cmp      edx, r8d
       jge      SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x0261
       lea      edx, [r8+0x01]
       shl      edx, 4
       movsxd   rdx, edx
       vmovups  zmmword ptr [rax+4*rdx], zmm3
 
G_M000_IG13:                ;; offset=0x0272
       inc      r13d
       cmp      r13d, r11d
       jge      G_M000_IG05
 
G_M000_IG14:                ;; offset=0x027E
       mov      ebx, dword ptr [rbp+0x10]
       mov      rdi, qword ptr [rbp-0x30]
       jmp      G_M000_IG07
 
; Total bytes of code 650

