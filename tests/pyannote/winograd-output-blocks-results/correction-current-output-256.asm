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
       mov      rdi, 0x72F65D54B018
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
       mov      rdi, 0x72F65D54B01C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
       lea      eax, [8*rax+0x08]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1D8]
       vmovups  ymm0, ymmword ptr [rbp-0x1D0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG09:                ;; offset=0x03FC
       mov      rdi, 0x72F65D54B020
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
       mov      rdi, 0x72F65D54B024
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
       mov      rdi, 0x72F65D54B028
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG16:                ;; offset=0x04A3
       vzeroupper 
       add      rsp, 576
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x04AF
       mov      rdi, 0x72F65D54B02C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG09
 
; Total bytes of code 1219

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
       mov      rdi, 0x72F65D54B018
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
       mov      rdi, 0x72F65D54B01C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
       lea      eax, [8*rax+0x08]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1D8]
       vmovups  ymm0, ymmword ptr [rbp-0x1D0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG09:                ;; offset=0x03FC
       mov      rdi, 0x72F65D54B020
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
       mov      rdi, 0x72F65D54B024
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
       mov      rdi, 0x72F65D54B028
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG16:                ;; offset=0x04A3
       vzeroupper 
       add      rsp, 576
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x04AF
       mov      rdi, 0x72F65D54B02C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG09
 
; Total bytes of code 1219

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 32228

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
       add      r14d, 8
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
       lea      edx, [8*r13]
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
       vmovups  ymm0, ymmword ptr [r12]
       mov      eax, dword ptr [rbp-0x38]
       lea      ebx, [4*rax]
       movsxd   rbx, ebx
       vmovups  ymm1, ymmword ptr [r12+4*rbx]
       vaddps   ymm0, ymm0, ymm1
       lea      ebx, [8*rax]
       movsxd   rbx, ebx
       vmovups  ymm2, ymmword ptr [r12+4*rbx]
       vaddps   ymm0, ymm0, ymm2
       vsubps   ymm1, ymm1, ymm2
       lea      ebx, [rax+2*rax]
       lea      r15d, [4*rbx]
       movsxd   r15, r15d
       vsubps   ymm1, ymm1, ymmword ptr [r12+4*r15]
       movsxd   r15, eax
       vmovups  ymm2, ymmword ptr [r12+4*r15]
       lea      r15d, [rax+4*rax]
       movsxd   rdi, r15d
       vmovups  ymm3, ymmword ptr [r12+4*rdi]
       vaddps   ymm2, ymm2, ymm3
       lea      edi, [rax+8*rax]
       movsxd   rdi, edi
       vmovups  ymm4, ymmword ptr [r12+4*rdi]
       vaddps   ymm2, ymm2, ymm4
       vsubps   ymm3, ymm3, ymm4
       imul     edi, eax, 13
       movsxd   rdi, edi
       vsubps   ymm3, ymm3, ymmword ptr [r12+4*rdi]
       lea      edi, [rax+rax]
       movsxd   rdi, edi
       vmovups  ymm4, ymmword ptr [r12+4*rdi]
       lea      edi, [rbx+rbx]
       movsxd   rdi, edi
       vmovups  ymm5, ymmword ptr [r12+4*rdi]
       vaddps   ymm4, ymm4, ymm5
       add      r15d, r15d
       movsxd   rdi, r15d
       vmovups  ymm6, ymmword ptr [r12+4*rdi]
       vaddps   ymm4, ymm6, ymm4
       vsubps   ymm5, ymm5, ymm6
       imul     edi, eax, 14
       movsxd   rdi, edi
       vsubps   ymm5, ymm5, ymmword ptr [r12+4*rdi]
       movsxd   rdi, ebx
 
G_M000_IG08:                ;; offset=0x0173
       vmovups  ymm6, ymmword ptr [r12+4*rdi]
       lea      edi, [8*rax]
       sub      edi, eax
       movsxd   rdi, edi
       vmovups  ymm7, ymmword ptr [r12+4*rdi]
       vaddps   ymm6, ymm6, ymm7
       imul     edi, eax, 11
       movsxd   rdi, edi
       vmovups  ymm8, ymmword ptr [r12+4*rdi]
       vaddps   ymm6, ymm6, ymm8
       vsubps   ymm7, ymm7, ymm8
       mov      edi, eax
       shl      edi, 4
       sub      edi, eax
       movsxd   rax, edi
       vsubps   ymm7, ymm7, ymmword ptr [r12+4*rax]
       vaddps   ymm0, ymm2, ymm0
       vaddps   ymm0, ymm0, ymm4
       vsubps   ymm2, ymm2, ymm4
       vsubps   ymm2, ymm2, ymm6
       vaddps   ymm1, ymm1, ymm3
       vaddps   ymm1, ymm1, ymm5
       vsubps   ymm3, ymm3, ymm5
       vsubps   ymm3, ymm3, ymm7
       mov      edi, dword ptr [rbp-0x3C]
       mov      eax, edi
       imul     eax, r8d
       add      eax, edx
       shl      eax, 3
       cdqe     
       mov      r15d, dword ptr [rbp-0x34]
       mov      ebx, r14d
       imul     ebx, r15d
       movsxd   rbx, ebx
       shl      rbx, 2
       add      rbx, rsi
       lea      rax, [rbx+4*rax]
       vmovups  ymmword ptr [rax], ymm0
       inc      edx
       cmp      edx, r8d
       jge      SHORT G_M000_IG10
 
G_M000_IG09:                ;; offset=0x0209
       vmovups  ymmword ptr [rax+0x20], ymm2
 
G_M000_IG10:                ;; offset=0x020E
       inc      edi
       cmp      edi, ecx
       jge      SHORT G_M000_IG13
 
G_M000_IG11:                ;; offset=0x0214
       lea      edi, [8*r8]
       movsxd   rbx, edi
       vmovups  ymmword ptr [rax+4*rbx], ymm1
       cmp      edx, r8d
       jge      SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x0229
       add      edi, 8
       movsxd   rdx, edi
       vmovups  ymmword ptr [rax+4*rdx], ymm3
 
G_M000_IG13:                ;; offset=0x0234
       inc      r13d
       cmp      r13d, r11d
       jge      G_M000_IG05
 
G_M000_IG14:                ;; offset=0x0240
       mov      ebx, dword ptr [rbp+0x10]
       mov      rdi, qword ptr [rbp-0x30]
       jmp      G_M000_IG07
 
; Total bytes of code 588

