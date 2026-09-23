; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel256(ptr,ptr,ptr,int,int,int,int,int,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x400
       lea      rbp, [rsp+0x400]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x310], xmm8
       vmovdqa  xmmword ptr [rbp-0x300], xmm8
       mov      rax, -672
       vmovdqa  xmmword ptr [rbp+rax-0x50], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      dword ptr [rbp-0x50], eax
       mov      qword ptr [rbp-0x30], rdi
       mov      qword ptr [rbp-0x38], rsi
       mov      qword ptr [rbp-0x40], rdx
       mov      dword ptr [rbp-0x44], ecx
       mov      dword ptr [rbp-0x48], r8d
       mov      dword ptr [rbp-0x4C], r9d
 
G_M000_IG02:                ;; offset=0x0061
       mov      dword ptr [rbp-0x3F8], 0x3E8
       mov      eax, dword ptr [rbp-0x4C]
       add      eax, 2
       mov      dword ptr [rbp-0x50], eax
       mov      eax, dword ptr [rbp+0x10]
       add      eax, 2
       mov      dword ptr [rbp-0x54], eax
       mov      eax, dword ptr [rbp+0x20]
       imul     eax, dword ptr [rbp+0x28]
       mov      dword ptr [rbp-0x58], eax
       mov      eax, dword ptr [rbp-0x58]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x58]
       sar      eax, 3
       shl      eax, 3
       mov      dword ptr [rbp-0x5C], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x60], eax
       jmp      G_M000_IG51
 
G_M000_IG03:                ;; offset=0x00A6
       xor      eax, eax
       mov      dword ptr [rbp-0x64], eax
       jmp      G_M000_IG48
 
G_M000_IG04:                ;; offset=0x00B0
       xor      eax, eax
       mov      dword ptr [rbp-0x68], eax
       jmp      G_M000_IG26
 
G_M000_IG05:                ;; offset=0x00BA
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x90], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0xB0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0xD0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0xF0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x110], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x130], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x150], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x170], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x190], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x1B0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x1D0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x1F0], ymm0
       mov      eax, dword ptr [rbp-0x60]
       imul     eax, dword ptr [rbp-0x44]
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x38]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x1F8], rax
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 3
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1F8]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x200], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x204], eax
       jmp      G_M000_IG17
 
G_M000_IG06:                ;; offset=0x018F
       xor      eax, eax
       mov      dword ptr [rbp-0x208], eax
       jmp      G_M000_IG14
 
G_M000_IG07:                ;; offset=0x019C
       xor      eax, eax
       mov      dword ptr [rbp-0x20C], eax
       jmp      G_M000_IG11
 
G_M000_IG08:                ;; offset=0x01A9
       mov      rdi, 0x761AD7AFB980
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x1F8]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x230], ymm0
       mov      rax, qword ptr [rbp-0x200]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x250], ymm0
       mov      eax, dword ptr [rbp-0x204]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x204]
       sar      eax, 3
       imul     eax, dword ptr [rbp-0x50]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x208]
       imul     eax, dword ptr [rbp-0x54]
       mov      ecx, dword ptr [rbp-0x68]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x20C]
       shl      eax, 3
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x30]
       mov      ecx, dword ptr [rbp-0x204]
       mov      edx, dword ptr [rbp-0x204]
       sar      edx, 31
       and      edx, 7
       add      edx, dword ptr [rbp-0x204]
       and      edx, -8
       sub      ecx, edx
       movsxd   rcx, ecx
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x318], rax
       mov      rax, qword ptr [rbp-0x318]
       vbroadcastss ymm0, dword ptr [rax]
       vmovups  ymmword ptr [rbp-0x350], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x350]
       vmovups  ymm1, ymmword ptr [rbp-0x90]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x230]
       vmovups  ymmword ptr [rbp-0x90], ymm1
       vmovups  ymm0, ymmword ptr [rbp-0x350]
       vmovups  ymm1, ymmword ptr [rbp-0xB0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x250]
       vmovups  ymmword ptr [rbp-0xB0], ymm1
       mov      eax, dword ptr [rbp+0x18]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x318]
       vbroadcastss ymm0, dword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x370], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x370]
       vmovups  ymm1, ymmword ptr [rbp-0xD0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x230]
       vmovups  ymmword ptr [rbp-0xD0], ymm1
       vmovups  ymm0, ymmword ptr [rbp-0x370]
       vmovups  ymm1, ymmword ptr [rbp-0xF0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x250]
 
G_M000_IG09:                ;; offset=0x02FE
       vmovups  ymmword ptr [rbp-0xF0], ymm1
       mov      eax, dword ptr [rbp+0x18]
       add      eax, eax
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x318]
       vbroadcastss ymm0, dword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x390], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x390]
       vmovups  ymm1, ymmword ptr [rbp-0x110]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x230]
       vmovups  ymmword ptr [rbp-0x110], ymm1
       vmovups  ymm0, ymmword ptr [rbp-0x390]
       vmovups  ymm1, ymmword ptr [rbp-0x130]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x250]
       vmovups  ymmword ptr [rbp-0x130], ymm1
       mov      eax, dword ptr [rbp+0x18]
       lea      eax, [rax+2*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x318]
       vbroadcastss ymm0, dword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x3B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x3B0]
       vmovups  ymm1, ymmword ptr [rbp-0x150]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x230]
       vmovups  ymmword ptr [rbp-0x150], ymm1
       vmovups  ymm0, ymmword ptr [rbp-0x3B0]
       vmovups  ymm1, ymmword ptr [rbp-0x170]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x250]
       vmovups  ymmword ptr [rbp-0x170], ymm1
       mov      eax, dword ptr [rbp+0x18]
       shl      eax, 2
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x318]
       vbroadcastss ymm0, dword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x3D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x3D0]
       vmovups  ymm1, ymmword ptr [rbp-0x190]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x230]
       vmovups  ymmword ptr [rbp-0x190], ymm1
       vmovups  ymm0, ymmword ptr [rbp-0x3D0]
       vmovups  ymm1, ymmword ptr [rbp-0x1B0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x250]
       vmovups  ymmword ptr [rbp-0x1B0], ymm1
       mov      eax, dword ptr [rbp+0x18]
       lea      eax, [rax+4*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x318]
       vbroadcastss ymm0, dword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x3F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x3F0]
       vmovups  ymm1, ymmword ptr [rbp-0x1D0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x230]
       vmovups  ymmword ptr [rbp-0x1D0], ymm1
       vmovups  ymm0, ymmword ptr [rbp-0x3F0]
       vmovups  ymm1, ymmword ptr [rbp-0x1F0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x250]
       vmovups  ymmword ptr [rbp-0x1F0], ymm1
 
G_M000_IG10:                ;; offset=0x048D
       mov      rax, qword ptr [rbp-0x1F8]
       add      rax, 32
       mov      qword ptr [rbp-0x1F8], rax
       mov      rax, qword ptr [rbp-0x200]
       add      rax, 32
       mov      qword ptr [rbp-0x200], rax
       mov      eax, dword ptr [rbp-0x20C]
       inc      eax
       mov      dword ptr [rbp-0x20C], eax
 
G_M000_IG11:                ;; offset=0x04BF
       mov      eax, dword ptr [rbp-0x3F8]
       dec      eax
       mov      dword ptr [rbp-0x3F8], eax
       cmp      dword ptr [rbp-0x3F8], 0
       jg       SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x04D6
       lea      rdi, [rbp-0x3F8]
       mov      esi, 492
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG13:                ;; offset=0x04E7
       cmp      dword ptr [rbp-0x20C], 3
       jl       G_M000_IG08
       mov      rdi, 0x761AD7AFB984
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x208]
       inc      eax
       mov      dword ptr [rbp-0x208], eax
 
G_M000_IG14:                ;; offset=0x0511
       mov      eax, dword ptr [rbp-0x3F8]
       dec      eax
       mov      dword ptr [rbp-0x3F8], eax
       cmp      dword ptr [rbp-0x3F8], 0
       jg       SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x0528
       lea      rdi, [rbp-0x3F8]
       mov      esi, 506
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG16:                ;; offset=0x0539
       cmp      dword ptr [rbp-0x208], 3
       jl       G_M000_IG07
       mov      rdi, 0x761AD7AFB988
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x204]
       inc      eax
       mov      dword ptr [rbp-0x204], eax
 
G_M000_IG17:                ;; offset=0x0563
       mov      eax, dword ptr [rbp-0x3F8]
       dec      eax
       mov      dword ptr [rbp-0x3F8], eax
       cmp      dword ptr [rbp-0x3F8], 0
       jg       SHORT G_M000_IG19
 
G_M000_IG18:                ;; offset=0x057A
       lea      rdi, [rbp-0x3F8]
       mov      esi, 520
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG19:                ;; offset=0x058B
       mov      eax, dword ptr [rbp-0x204]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG06
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x90]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 8
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG20
       mov      rdi, 0x761AD7AFB98C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0xB0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG20:                ;; offset=0x0620
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       lea      eax, [8*rax+0x08]
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0xD0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 8
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG21
       mov      rdi, 0x761AD7AFB990
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       lea      eax, [8*rax+0x08]
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0xF0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG21:                ;; offset=0x06AE
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       lea      eax, [8*rax+0x10]
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x110]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 8
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG22
       mov      rdi, 0x761AD7AFB994
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       lea      eax, [8*rax+0x10]
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x130]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG22:                ;; offset=0x073C
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       lea      eax, [8*rax+0x18]
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x150]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 8
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG23
       mov      rdi, 0x761AD7AFB998
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       lea      eax, [8*rax+0x18]
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x170]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG23:                ;; offset=0x07CA
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       lea      eax, [8*rax+0x20]
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 8
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG24
       mov      rdi, 0x761AD7AFB99C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       lea      eax, [8*rax+0x20]
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x1B0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG24:                ;; offset=0x0858
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       lea      eax, [8*rax+0x28]
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x1D0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 8
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG25
       mov      rdi, 0x761AD7AFB9A0
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       lea      eax, [8*rax+0x28]
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x1F0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG25:                ;; offset=0x08E6
       mov      rdi, 0x761AD7AFB9A4
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x68]
       add      eax, 6
       mov      dword ptr [rbp-0x68], eax
 
G_M000_IG26:                ;; offset=0x08FE
       mov      eax, dword ptr [rbp-0x68]
       add      eax, 6
       cmp      eax, dword ptr [rbp+0x28]
       jg       G_M000_IG45
       mov      eax, dword ptr [rbp-0x3F8]
       dec      eax
       mov      dword ptr [rbp-0x3F8], eax
       cmp      dword ptr [rbp-0x3F8], 0
       jg       SHORT G_M000_IG28
 
G_M000_IG27:                ;; offset=0x0924
       lea      rdi, [rbp-0x3F8]
       mov      esi, 973
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG28:                ;; offset=0x0935
       mov      eax, dword ptr [rbp-0x64]
       imul     eax, dword ptr [rbp+0x28]
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x06]
       cmp      eax, dword ptr [rbp-0x5C]
       jle      G_M000_IG05
       mov      rdi, 0x761AD7AFB9A8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG45
 
G_M000_IG29:                ;; offset=0x0960
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x270], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x290], ymm0
       mov      eax, dword ptr [rbp-0x60]
       imul     eax, dword ptr [rbp-0x44]
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x38]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x298], rax
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 3
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x298]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x2A0], rax
       mov      eax, dword ptr [rbp-0x64]
       imul     eax, dword ptr [rbp+0x28]
       add      eax, dword ptr [rbp-0x68]
       cmp      eax, dword ptr [rbp-0x5C]
       setl     al
       movzx    rax, al
       mov      dword ptr [rbp-0x2A4], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x2A8], eax
       jmp      G_M000_IG41
 
G_M000_IG30:                ;; offset=0x09D6
       xor      eax, eax
       mov      dword ptr [rbp-0x2AC], eax
       jmp      G_M000_IG38
 
G_M000_IG31:                ;; offset=0x09E3
       xor      eax, eax
       mov      dword ptr [rbp-0x2B0], eax
       jmp      G_M000_IG35
 
G_M000_IG32:                ;; offset=0x09F0
       mov      eax, dword ptr [rbp-0x2A8]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x2A8]
       sar      eax, 3
       imul     eax, dword ptr [rbp-0x50]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x2AC]
       imul     eax, dword ptr [rbp-0x54]
       mov      ecx, dword ptr [rbp-0x68]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x2B0]
       mov      ecx, dword ptr [rbp-0x2A8]
       mov      edx, dword ptr [rbp-0x2A8]
       sar      edx, 31
       and      edx, 7
       add      edx, dword ptr [rbp-0x2A8]
       and      edx, -8
       sub      ecx, edx
       lea      eax, [rcx+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       vbroadcastss ymm0, dword ptr [rcx+4*rax]
       vmovups  ymmword ptr [rbp-0x2D0], ymm0
       mov      rax, qword ptr [rbp-0x298]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x2F0], ymm0
       mov      rax, qword ptr [rbp-0x2A0]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x310], ymm0
       cmp      dword ptr [rbp-0x2A4], 0
       je       SHORT G_M000_IG33
       vmovups  ymm0, ymmword ptr [rbp-0x2D0]
       vmovups  ymm1, ymmword ptr [rbp-0x270]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x2F0]
       vmovups  ymmword ptr [rbp-0x270], ymm1
       vmovups  ymm0, ymmword ptr [rbp-0x2D0]
       vmovups  ymm1, ymmword ptr [rbp-0x290]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x310]
       vmovups  ymmword ptr [rbp-0x290], ymm1
       jmp      SHORT G_M000_IG34
 
G_M000_IG33:                ;; offset=0x0AD2
       mov      rdi, 0x761AD7AFB9AC
       call     CORINFO_HELP_COUNTPROFILE32
       vmovups  ymm0, ymmword ptr [rbp-0x2D0]
       vmulps   ymm0, ymm0, ymmword ptr [rbp-0x2F0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x270]
       vmovups  ymmword ptr [rbp-0x270], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x2D0]
       vmulps   ymm0, ymm0, ymmword ptr [rbp-0x310]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x290]
       vmovups  ymmword ptr [rbp-0x290], ymm0
 
G_M000_IG34:                ;; offset=0x0B21
       mov      rdi, 0x761AD7AFB9B0
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x298]
       add      rax, 32
       mov      qword ptr [rbp-0x298], rax
       mov      rax, qword ptr [rbp-0x2A0]
       add      rax, 32
       mov      qword ptr [rbp-0x2A0], rax
       mov      eax, dword ptr [rbp-0x2B0]
       inc      eax
       mov      dword ptr [rbp-0x2B0], eax
 
G_M000_IG35:                ;; offset=0x0B62
       mov      eax, dword ptr [rbp-0x3F8]
       dec      eax
       mov      dword ptr [rbp-0x3F8], eax
       cmp      dword ptr [rbp-0x3F8], 0
       jg       SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x0B79
       lea      rdi, [rbp-0x3F8]
       mov      esi, 0x4CD
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG37:                ;; offset=0x0B8A
       cmp      dword ptr [rbp-0x2B0], 3
       jl       G_M000_IG32
       mov      rdi, 0x761AD7AFB9B4
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x2AC]
       inc      eax
       mov      dword ptr [rbp-0x2AC], eax
 
G_M000_IG38:                ;; offset=0x0BB4
       mov      eax, dword ptr [rbp-0x3F8]
       dec      eax
       mov      dword ptr [rbp-0x3F8], eax
       cmp      dword ptr [rbp-0x3F8], 0
       jg       SHORT G_M000_IG40
 
G_M000_IG39:                ;; offset=0x0BCB
       lea      rdi, [rbp-0x3F8]
       mov      esi, 0x4DB
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG40:                ;; offset=0x0BDC
       cmp      dword ptr [rbp-0x2AC], 3
       jl       G_M000_IG31
       mov      rdi, 0x761AD7AFB9B8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x2A8]
       inc      eax
       mov      dword ptr [rbp-0x2A8], eax
 
G_M000_IG41:                ;; offset=0x0C06
       mov      eax, dword ptr [rbp-0x3F8]
       dec      eax
       mov      dword ptr [rbp-0x3F8], eax
       cmp      dword ptr [rbp-0x3F8], 0
       jg       SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x0C1D
       lea      rdi, [rbp-0x3F8]
       mov      esi, 0x4E9
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG43:                ;; offset=0x0C2E
       mov      eax, dword ptr [rbp-0x2A8]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG30
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x270]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 8
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG44
       mov      rdi, 0x761AD7AFB9BC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 7
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 3
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  ymm0, ymmword ptr [rbp-0x290]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG44:                ;; offset=0x0CC3
       mov      rdi, 0x761AD7AFB9C0
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x68]
       inc      eax
       mov      dword ptr [rbp-0x68], eax
 
G_M000_IG45:                ;; offset=0x0CDA
       mov      eax, dword ptr [rbp-0x3F8]
       dec      eax
       mov      dword ptr [rbp-0x3F8], eax
       cmp      dword ptr [rbp-0x3F8], 0
       jg       SHORT G_M000_IG47
 
G_M000_IG46:                ;; offset=0x0CF1
       lea      rdi, [rbp-0x3F8]
       mov      esi, 0x53B
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG47:                ;; offset=0x0D02
       mov      eax, dword ptr [rbp-0x68]
       cmp      eax, dword ptr [rbp+0x28]
       jl       G_M000_IG29
       mov      rdi, 0x761AD7AFB9C4
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x64]
       inc      eax
       mov      dword ptr [rbp-0x64], eax
 
G_M000_IG48:                ;; offset=0x0D25
       mov      eax, dword ptr [rbp-0x3F8]
       dec      eax
       mov      dword ptr [rbp-0x3F8], eax
       cmp      dword ptr [rbp-0x3F8], 0
       jg       SHORT G_M000_IG50
 
G_M000_IG49:                ;; offset=0x0D3C
       lea      rdi, [rbp-0x3F8]
       mov      esi, 0x54A
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG50:                ;; offset=0x0D4D
       mov      eax, dword ptr [rbp-0x64]
       cmp      eax, dword ptr [rbp+0x20]
       jl       G_M000_IG04
       mov      rdi, 0x761AD7AFB9C8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 16
       mov      dword ptr [rbp-0x60], eax
 
G_M000_IG51:                ;; offset=0x0D71
       mov      eax, dword ptr [rbp-0x3F8]
       dec      eax
       mov      dword ptr [rbp-0x3F8], eax
       cmp      dword ptr [rbp-0x3F8], 0
       jg       SHORT G_M000_IG53
 
G_M000_IG52:                ;; offset=0x0D88
       lea      rdi, [rbp-0x3F8]
       mov      esi, 0x55A
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG53:                ;; offset=0x0D99
       mov      eax, dword ptr [rbp-0x60]
       cmp      eax, dword ptr [rbp-0x48]
       jl       G_M000_IG03
       mov      rdi, 0x761AD7AFB9CC
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG54:                ;; offset=0x0DB5
       vzeroupper 
       add      rsp, 0x400
       pop      rbp
       ret      
 
; Total bytes of code 3521

