; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel512(ptr,ptr,ptr,int,int,int,int,int,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x780
       lea      rbp, [rsp+0x780]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa32 xmmword ptr [rbp-0x5B0], xmm8
       vmovdqa32 xmmword ptr [rbp-0x5A0], xmm8
       mov      rax, -0x540
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
 
G_M000_IG02:                ;; offset=0x005F
       mov      dword ptr [rbp-0x778], 0x3E8
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
 
G_M000_IG03:                ;; offset=0x00A4
       xor      eax, eax
       mov      dword ptr [rbp-0x64], eax
       jmp      G_M000_IG48
 
G_M000_IG04:                ;; offset=0x00AE
       xor      eax, eax
       mov      dword ptr [rbp-0x68], eax
       jmp      G_M000_IG26
 
G_M000_IG05:                ;; offset=0x00B8
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0xB0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0xF0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x130], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x170], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x1B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x1F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x230], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x270], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x2B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x2F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x330], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x370], zmm0
       mov      eax, dword ptr [rbp-0x60]
       imul     eax, dword ptr [rbp-0x44]
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x38]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x378], rax
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 4
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x378]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x380], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x384], eax
       jmp      G_M000_IG17
 
G_M000_IG06:                ;; offset=0x01A5
       xor      eax, eax
       mov      dword ptr [rbp-0x388], eax
       jmp      G_M000_IG14
 
G_M000_IG07:                ;; offset=0x01B2
       xor      eax, eax
       mov      dword ptr [rbp-0x38C], eax
       jmp      G_M000_IG11
 
G_M000_IG08:                ;; offset=0x01BF
       mov      rdi, 0x7C353D921F38
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x378]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x3F0], zmm0
       mov      rax, qword ptr [rbp-0x380]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x430], zmm0
       mov      eax, dword ptr [rbp-0x384]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x384]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x50]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x388]
       imul     eax, dword ptr [rbp-0x54]
       mov      ecx, dword ptr [rbp-0x68]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x38C]
       shl      eax, 4
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x30]
       mov      ecx, dword ptr [rbp-0x384]
       mov      edx, dword ptr [rbp-0x384]
       sar      edx, 31
       and      edx, 15
       add      edx, dword ptr [rbp-0x384]
       and      edx, -16
       sub      ecx, edx
       movsxd   rcx, ecx
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x5B8], rax
       mov      rax, qword ptr [rbp-0x5B8]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x630], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x630]
       vmovups  zmm1, zmmword ptr [rbp-0xB0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x3F0]
       vmovups  zmmword ptr [rbp-0xB0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x630]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x430]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       mov      eax, dword ptr [rbp+0x18]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x5B8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x670], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x670]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x3F0]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x670]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x430]
 
G_M000_IG09:                ;; offset=0x033C
       vmovups  zmmword ptr [rbp-0x170], zmm1
       mov      eax, dword ptr [rbp+0x18]
       add      eax, eax
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x5B8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x6B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x6B0]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x3F0]
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x6B0]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x430]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       mov      eax, dword ptr [rbp+0x18]
       lea      eax, [rax+2*rax]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x5B8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x6F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x6F0]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x3F0]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x6F0]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x430]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       mov      eax, dword ptr [rbp+0x18]
       shl      eax, 2
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x5B8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x730], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x730]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x3F0]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x730]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x430]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       mov      eax, dword ptr [rbp+0x18]
       lea      eax, [rax+4*rax]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x5B8]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x770], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x770]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x3F0]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x770]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x430]
       vmovups  zmmword ptr [rbp-0x370], zmm1
 
G_M000_IG10:                ;; offset=0x0511
       mov      rax, qword ptr [rbp-0x378]
       add      rax, 64
       mov      qword ptr [rbp-0x378], rax
       mov      rax, qword ptr [rbp-0x380]
       add      rax, 64
       mov      qword ptr [rbp-0x380], rax
       mov      eax, dword ptr [rbp-0x38C]
       inc      eax
       mov      dword ptr [rbp-0x38C], eax
 
G_M000_IG11:                ;; offset=0x0543
       mov      eax, dword ptr [rbp-0x778]
       dec      eax
       mov      dword ptr [rbp-0x778], eax
       cmp      dword ptr [rbp-0x778], 0
       jg       SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x055A
       lea      rdi, [rbp-0x778]
       mov      esi, 503
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG13:                ;; offset=0x056B
       cmp      dword ptr [rbp-0x38C], 3
       jl       G_M000_IG08
       mov      rdi, 0x7C353D921F3C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x388]
       inc      eax
       mov      dword ptr [rbp-0x388], eax
 
G_M000_IG14:                ;; offset=0x0595
       mov      eax, dword ptr [rbp-0x778]
       dec      eax
       mov      dword ptr [rbp-0x778], eax
       cmp      dword ptr [rbp-0x778], 0
       jg       SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x05AC
       lea      rdi, [rbp-0x778]
       mov      esi, 517
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG16:                ;; offset=0x05BD
       cmp      dword ptr [rbp-0x388], 3
       jl       G_M000_IG07
       mov      rdi, 0x7C353D921F40
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x384]
       inc      eax
       mov      dword ptr [rbp-0x384], eax
 
G_M000_IG17:                ;; offset=0x05E7
       mov      eax, dword ptr [rbp-0x778]
       dec      eax
       mov      dword ptr [rbp-0x778], eax
       cmp      dword ptr [rbp-0x778], 0
       jg       SHORT G_M000_IG19
 
G_M000_IG18:                ;; offset=0x05FE
       lea      rdi, [rbp-0x778]
       mov      esi, 531
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG19:                ;; offset=0x060F
       mov      eax, dword ptr [rbp-0x384]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG06
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0xB0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 16
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG20
       mov      rdi, 0x7C353D921F44
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0xF0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG20:                ;; offset=0x06AC
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x01]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x130]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 16
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG21
       mov      rdi, 0x7C353D921F48
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x01]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x170]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG21:                ;; offset=0x0742
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x02]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x1B0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 16
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG22
       mov      rdi, 0x7C353D921F4C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x02]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x1F0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG22:                ;; offset=0x07D8
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x03]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x230]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 16
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG23
       mov      rdi, 0x7C353D921F50
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x03]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x270]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG23:                ;; offset=0x086E
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x04]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x2B0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 16
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG24
       mov      rdi, 0x7C353D921F54
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x04]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG24:                ;; offset=0x0904
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x05]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x330]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 16
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG25
       mov      rdi, 0x7C353D921F58
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x05]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x370]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG25:                ;; offset=0x099A
       mov      rdi, 0x7C353D921F5C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x68]
       add      eax, 6
       mov      dword ptr [rbp-0x68], eax
 
G_M000_IG26:                ;; offset=0x09B2
       mov      eax, dword ptr [rbp-0x68]
       add      eax, 6
       cmp      eax, dword ptr [rbp+0x28]
       jg       G_M000_IG45
       mov      eax, dword ptr [rbp-0x778]
       dec      eax
       mov      dword ptr [rbp-0x778], eax
       cmp      dword ptr [rbp-0x778], 0
       jg       SHORT G_M000_IG28
 
G_M000_IG27:                ;; offset=0x09D8
       lea      rdi, [rbp-0x778]
       mov      esi, 0x3F6
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG28:                ;; offset=0x09E9
       mov      eax, dword ptr [rbp-0x64]
       imul     eax, dword ptr [rbp+0x28]
       mov      ecx, dword ptr [rbp-0x68]
       lea      eax, [rax+rcx+0x06]
       cmp      eax, dword ptr [rbp-0x5C]
       jle      G_M000_IG05
       mov      rdi, 0x7C353D921F60
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG45
 
G_M000_IG29:                ;; offset=0x0A14
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x470], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x4B0], zmm0
       mov      eax, dword ptr [rbp-0x60]
       imul     eax, dword ptr [rbp-0x44]
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x38]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x4B8], rax
       mov      eax, dword ptr [rbp-0x44]
       shl      eax, 4
       lea      eax, [rax+8*rax]
       cdqe     
       mov      rcx, qword ptr [rbp-0x4B8]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x4C0], rax
       mov      eax, dword ptr [rbp-0x64]
       imul     eax, dword ptr [rbp+0x28]
       add      eax, dword ptr [rbp-0x68]
       cmp      eax, dword ptr [rbp-0x5C]
       setl     al
       movzx    rax, al
       mov      dword ptr [rbp-0x4C4], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x4C8], eax
       jmp      G_M000_IG41
 
G_M000_IG30:                ;; offset=0x0A8E
       xor      eax, eax
       mov      dword ptr [rbp-0x4CC], eax
       jmp      G_M000_IG38
 
G_M000_IG31:                ;; offset=0x0A9B
       xor      eax, eax
       mov      dword ptr [rbp-0x4D0], eax
       jmp      G_M000_IG35
 
G_M000_IG32:                ;; offset=0x0AA8
       mov      eax, dword ptr [rbp-0x4C8]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x4C8]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x50]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x4CC]
       imul     eax, dword ptr [rbp-0x54]
       mov      ecx, dword ptr [rbp-0x68]
       imul     ecx, dword ptr [rbp+0x18]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x4D0]
       shl      eax, 4
       mov      ecx, dword ptr [rbp-0x4C8]
       mov      edx, dword ptr [rbp-0x4C8]
       sar      edx, 31
       and      edx, 15
       add      edx, dword ptr [rbp-0x4C8]
       and      edx, -16
       sub      ecx, edx
       add      eax, ecx
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       vbroadcastss zmm0, dword ptr [rcx+4*rax]
       vmovups  zmmword ptr [rbp-0x530], zmm0
       mov      rax, qword ptr [rbp-0x4B8]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x570], zmm0
       mov      rax, qword ptr [rbp-0x4C0]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x5B0], zmm0
       cmp      dword ptr [rbp-0x4C4], 0
       je       SHORT G_M000_IG33
       vmovups  zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x570]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmm1, zmmword ptr [rbp-0x4B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x5B0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm1
       jmp      SHORT G_M000_IG34
 
G_M000_IG33:                ;; offset=0x0BA5
       mov      rdi, 0x7C353D921F64
       call     CORINFO_HELP_COUNTPROFILE32
       vmovups  zmm0, zmmword ptr [rbp-0x530]
       vmulps   zmm0, zmm0, zmmword ptr [rbp-0x570]
       vaddps   zmm0, zmm0, zmmword ptr [rbp-0x470]
       vmovups  zmmword ptr [rbp-0x470], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x530]
       vmulps   zmm0, zmm0, zmmword ptr [rbp-0x5B0]
       vaddps   zmm0, zmm0, zmmword ptr [rbp-0x4B0]
       vmovups  zmmword ptr [rbp-0x4B0], zmm0
 
G_M000_IG34:                ;; offset=0x0C04
       mov      rdi, 0x7C353D921F68
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x4B8]
       add      rax, 64
       mov      qword ptr [rbp-0x4B8], rax
       mov      rax, qword ptr [rbp-0x4C0]
       add      rax, 64
       mov      qword ptr [rbp-0x4C0], rax
       mov      eax, dword ptr [rbp-0x4D0]
       inc      eax
       mov      dword ptr [rbp-0x4D0], eax
 
G_M000_IG35:                ;; offset=0x0C45
       mov      eax, dword ptr [rbp-0x778]
       dec      eax
       mov      dword ptr [rbp-0x778], eax
       cmp      dword ptr [rbp-0x778], 0
       jg       SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x0C5C
       lea      rdi, [rbp-0x778]
       mov      esi, 0x4FC
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG37:                ;; offset=0x0C6D
       cmp      dword ptr [rbp-0x4D0], 3
       jl       G_M000_IG32
       mov      rdi, 0x7C353D921F6C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4CC]
       inc      eax
       mov      dword ptr [rbp-0x4CC], eax
 
G_M000_IG38:                ;; offset=0x0C97
       mov      eax, dword ptr [rbp-0x778]
       dec      eax
       mov      dword ptr [rbp-0x778], eax
       cmp      dword ptr [rbp-0x778], 0
       jg       SHORT G_M000_IG40
 
G_M000_IG39:                ;; offset=0x0CAE
       lea      rdi, [rbp-0x778]
       mov      esi, 0x50A
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG40:                ;; offset=0x0CBF
       cmp      dword ptr [rbp-0x4CC], 3
       jl       G_M000_IG31
       mov      rdi, 0x7C353D921F70
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C8]
       inc      eax
       mov      dword ptr [rbp-0x4C8], eax
 
G_M000_IG41:                ;; offset=0x0CE9
       mov      eax, dword ptr [rbp-0x778]
       dec      eax
       mov      dword ptr [rbp-0x778], eax
       cmp      dword ptr [rbp-0x778], 0
       jg       SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x0D00
       lea      rdi, [rbp-0x778]
       mov      esi, 0x518
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG43:                ;; offset=0x0D11
       mov      eax, dword ptr [rbp-0x4C8]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG30
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x470]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 16
       cmp      eax, dword ptr [rbp-0x48]
       jge      SHORT G_M000_IG44
       mov      rdi, 0x7C353D921F74
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       sar      eax, 31
       and      eax, 15
       add      eax, dword ptr [rbp-0x60]
       sar      eax, 4
       inc      eax
       imul     eax, dword ptr [rbp-0x58]
       mov      ecx, dword ptr [rbp-0x64]
       imul     ecx, dword ptr [rbp+0x28]
       add      eax, ecx
       add      eax, dword ptr [rbp-0x68]
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       vmovups  zmm0, zmmword ptr [rbp-0x4B0]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG44:                ;; offset=0x0DAE
       mov      rdi, 0x7C353D921F78
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x68]
       inc      eax
       mov      dword ptr [rbp-0x68], eax
 
G_M000_IG45:                ;; offset=0x0DC5
       mov      eax, dword ptr [rbp-0x778]
       dec      eax
       mov      dword ptr [rbp-0x778], eax
       cmp      dword ptr [rbp-0x778], 0
       jg       SHORT G_M000_IG47
 
G_M000_IG46:                ;; offset=0x0DDC
       lea      rdi, [rbp-0x778]
       mov      esi, 0x56F
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG47:                ;; offset=0x0DED
       mov      eax, dword ptr [rbp-0x68]
       cmp      eax, dword ptr [rbp+0x28]
       jl       G_M000_IG29
       mov      rdi, 0x7C353D921F7C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x64]
       inc      eax
       mov      dword ptr [rbp-0x64], eax
 
G_M000_IG48:                ;; offset=0x0E10
       mov      eax, dword ptr [rbp-0x778]
       dec      eax
       mov      dword ptr [rbp-0x778], eax
       cmp      dword ptr [rbp-0x778], 0
       jg       SHORT G_M000_IG50
 
G_M000_IG49:                ;; offset=0x0E27
       lea      rdi, [rbp-0x778]
       mov      esi, 0x57E
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG50:                ;; offset=0x0E38
       mov      eax, dword ptr [rbp-0x64]
       cmp      eax, dword ptr [rbp+0x20]
       jl       G_M000_IG04
       mov      rdi, 0x7C353D921F80
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x60]
       add      eax, 32
       mov      dword ptr [rbp-0x60], eax
 
G_M000_IG51:                ;; offset=0x0E5C
       mov      eax, dword ptr [rbp-0x778]
       dec      eax
       mov      dword ptr [rbp-0x778], eax
       cmp      dword ptr [rbp-0x778], 0
       jg       SHORT G_M000_IG53
 
G_M000_IG52:                ;; offset=0x0E73
       lea      rdi, [rbp-0x778]
       mov      esi, 0x58E
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG53:                ;; offset=0x0E84
       mov      eax, dword ptr [rbp-0x60]
       cmp      eax, dword ptr [rbp-0x48]
       jl       G_M000_IG03
       mov      rdi, 0x7C353D921F84
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG54:                ;; offset=0x0EA0
       vzeroupper 
       add      rsp, 0x780
       pop      rbp
       ret      
 
; Total bytes of code 3756

