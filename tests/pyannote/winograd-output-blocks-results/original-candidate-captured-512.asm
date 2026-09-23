; Assembly listing for method KernelAccess:PrepareWinograd(System.ReadOnlySpan`1[float],int,int,int):float[] (Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier0 code
; rbp based frame
; partially interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 48
       lea      rbp, [rsp+0x30]
       xor      eax, eax
       mov      qword ptr [rbp-0x28], rax
       mov      bword ptr [rbp-0x10], rdi
       mov      qword ptr [rbp-0x08], rsi
       mov      dword ptr [rbp-0x14], edx
       mov      dword ptr [rbp-0x18], ecx
       mov      dword ptr [rbp-0x1C], r8d
 
G_M000_IG02:                ;; offset=0x0022
       mov      rsi, bword ptr [rbp-0x10]
       mov      rdx, qword ptr [rbp-0x08]
       mov      rax, 0x7A7CAD800190
       mov      rax, gword ptr [rax]
       mov      gword ptr [rbp-0x28], rax
       mov      rax, gword ptr [rbp-0x28]
       mov      ecx, dword ptr [rbp-0x14]
       mov      r8d, dword ptr [rbp-0x18]
       mov      r9d, dword ptr [rbp-0x1C]
       mov      rdi, gword ptr [rax+0x08]
       mov      rax, gword ptr [rbp-0x28]
       call     [rax+0x18]KernelAccess+PrepareCall:Invoke(System.ReadOnlySpan`1[float],int,int,int):float[]:this
       nop      
 
G_M000_IG03:                ;; offset=0x0056
       add      rsp, 48
       pop      rbp
       ret      
 
; Total bytes of code 92

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:PrepareWinograd(System.ReadOnlySpan`1[float],int,int,int):float[] (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier-0 switched to FullOpts code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100
; No PGO data
; 0 inlinees with PGO data; 2 single block inlinees; 3 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 24
       lea      rbp, [rsp+0x40]
       mov      qword ptr [rbp-0x38], 0x1EAA0F60
       mov      r13, rdi
       mov      r14d, esi
       mov      ebx, edx
       mov      r15d, ecx
 
G_M000_IG02:                ;; offset=0x0026
       mov      edi, ebx
       mov      esi, r15d
       mov      r9d, r8d
       mov      edx, 1
       mov      ecx, 1
       mov      r8d, 1
       call     [Lokad.Onnx.ConvBlockedSpatial:Geometry(int,int,int,int,int,int)]
       mov      edi, ebx
       imul     edi, r15d
       jo       G_M000_IG21
       imul     edi, edi, 9
       jo       G_M000_IG21
       cmp      edi, r14d
       jne      G_M000_IG18
       mov      rdi, r13
       mov      esi, r14d
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG12
 
G_M000_IG03:                ;; offset=0x0076
       imul     r12d, ebx, 16
       jo       G_M000_IG21
       imul     r12d, r15d
       jo       G_M000_IG21
       movsxd   rsi, r12d
       mov      rdi, 0x7A8492525DB8
       call     CORINFO_HELP_NEWARR_1_VC
       mov      gword ptr [rbp-0x30], rax
       test     dword ptr [rsp], esp
       sub      rsp, 48
       lea      rcx, [rsp]
       vxorps   ymm0, ymm0, ymm0
       vmovdqu  ymmword ptr [rcx], ymm0
       vmovdqu  xmmword ptr [rcx+0x20], xmm0
       xor      edx, edx
       vmovss   xmm0, dword ptr [reloc @RWD00]
       cmp      edx, r15d
       jge      G_M000_IG11
 
G_M000_IG04:                ;; offset=0x00CB
       xor      edi, edi
       cmp      edi, ebx
       jge      G_M000_IG10
 
G_M000_IG05:                ;; offset=0x00D5
       mov      esi, edx
       imul     esi, ebx
       add      esi, edi
       lea      esi, [rsi+8*rsi]
       mov      r8d, esi
       add      r8, 9
       mov      r9d, r14d
       cmp      r8, r9
       ja       G_M000_IG19
       lea      rsi, bword ptr [r13+4*rsi]
       xor      r8d, r8d
       align    [0 bytes for IG06]
 
G_M000_IG06:                ;; offset=0x00FA
       lea      r9d, [r8+2*r8]
       mov      r10d, r9d
       vmovss   xmm1, dword ptr [rsi+4*r10]
       lea      r10d, [r9+0x01]
       vmovss   xmm2, dword ptr [rsi+4*r10]
       add      r9d, 2
       vmovss   xmm3, dword ptr [rsi+4*r9]
       lea      r9d, [4*r8]
       mov      r10d, r9d
       vmovss   dword ptr [rcx+4*r10], xmm1
       lea      r10d, [r9+0x01]
       vaddss   xmm4, xmm1, xmm2
       vaddss   xmm4, xmm4, xmm3
       vmulss   xmm4, xmm4, xmm0
       vmovss   dword ptr [rcx+4*r10], xmm4
       lea      r10d, [r9+0x02]
       vsubss   xmm1, xmm1, xmm2
       vaddss   xmm1, xmm1, xmm3
       vmulss   xmm1, xmm1, xmm0
       vmovss   dword ptr [rcx+4*r10], xmm1
       add      r9d, 3
       vmovss   dword ptr [rcx+4*r9], xmm3
       inc      r8d
       cmp      r8d, 3
       jl       SHORT G_M000_IG06
 
G_M000_IG07:                ;; offset=0x016B
       xor      esi, esi
       align    [0 bytes for IG08]
 
G_M000_IG08:                ;; offset=0x016D
       vmovss   xmm1, dword ptr [rcx+4*rsi]
       lea      r8d, [rsi+0x04]
       mov      r9d, r8d
       vmovss   xmm2, dword ptr [rcx+4*r9]
       lea      r9d, [rsi+0x08]
       mov      r10d, r9d
       vmovss   xmm3, dword ptr [rcx+4*r10]
       mov      r10d, esi
       imul     r10d, ebx
       add      r10d, edi
       imul     r10d, r15d
       add      r10d, edx
       cmp      r10d, r12d
       jae      G_M000_IG20
       vmovss   dword ptr [rax+4*r10+0x10], xmm1
       imul     r8d, ebx
       add      r8d, edi
       imul     r8d, r15d
       add      r8d, edx
       cmp      r8d, r12d
       jae      G_M000_IG20
       vaddss   xmm4, xmm1, xmm2
       vaddss   xmm4, xmm4, xmm3
       vmulss   xmm4, xmm4, xmm0
       vmovss   dword ptr [rax+4*r8+0x10], xmm4
       imul     r9d, ebx
       add      r9d, edi
       imul     r9d, r15d
       add      r9d, edx
       cmp      r9d, r12d
       jae      G_M000_IG20
       mov      r8d, r9d
       vsubss   xmm1, xmm1, xmm2
       vaddss   xmm1, xmm1, xmm3
       vmulss   xmm1, xmm1, xmm0
       vmovss   dword ptr [rax+4*r8+0x10], xmm1
       lea      r8d, [rsi+0x0C]
       imul     r8d, ebx
       add      r8d, edi
       imul     r8d, r15d
       add      r8d, edx
       cmp      r8d, r12d
       jae      G_M000_IG20
       vmovss   dword ptr [rax+4*r8+0x10], xmm3
       inc      esi
       cmp      esi, 4
       jl       G_M000_IG08
 
G_M000_IG09:                ;; offset=0x0231
       inc      edi
       cmp      edi, ebx
       jl       G_M000_IG05
 
G_M000_IG10:                ;; offset=0x023B
       inc      edx
       cmp      edx, r15d
       jl       G_M000_IG04
 
G_M000_IG11:                ;; offset=0x0246
       lea      rdi, bword ptr [rax+0x10]
       mov      esi, r12d
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG15
 
G_M000_IG12:                ;; offset=0x0257
       xor      rax, rax
       cmp      qword ptr [rbp-0x38], 0x1EAA0F60
       je       SHORT G_M000_IG13
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG13:                ;; offset=0x0268
       nop      
 
G_M000_IG14:                ;; offset=0x0269
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG15:                ;; offset=0x0278
       mov      rax, gword ptr [rbp-0x30]
       cmp      qword ptr [rbp-0x38], 0x1EAA0F60
       je       SHORT G_M000_IG16
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG16:                ;; offset=0x028B
       nop      
 
G_M000_IG17:                ;; offset=0x028C
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG18:                ;; offset=0x029B
       mov      rdi, 0x7A8491C8CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0x10202
       mov      rsi, 0x7A84924B9F30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG19:                ;; offset=0x02D7
       call     [System.ThrowHelper:ThrowArgumentOutOfRangeException()]
       int3     
 
G_M000_IG20:                ;; offset=0x02DE
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
G_M000_IG21:                ;; offset=0x02E4
       call     CORINFO_HELP_OVERFLOW
       int3     
 
RWD00  	dd	3F000000h		;       0.5

; Total bytes of code 746

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
       mov      rdi, 0x7A84927580D0
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
       mov      rdi, 0x7A84927580D4
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
       mov      rdi, 0x7A84927580D8
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
       mov      rdi, 0x7A84927580DC
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
       mov      rdi, 0x7A84927580E0
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
       mov      rdi, 0x7A84927580E4
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
       mov      rdi, 0x7A84927580E8
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
       mov      rdi, 0x7A84927580EC
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
       mov      rdi, 0x7A84927580F0
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
       mov      rdi, 0x7A84927580F4
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
       mov      rdi, 0x7A84927580F8
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
       mov      rdi, 0x7A84927580FC
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
       mov      rdi, 0x7A8492758100
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
       mov      rdi, 0x7A8492758104
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
       mov      rdi, 0x7A8492758108
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
       mov      rdi, 0x7A849275810C
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
       mov      rdi, 0x7A8492758110
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
       mov      rdi, 0x7A8492758114
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
       mov      rdi, 0x7A8492758118
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
       mov      rdi, 0x7A849275811C
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG54:                ;; offset=0x0EA0
       vzeroupper 
       add      rsp, 0x780
       pop      rbp
       ret      
 
; Total bytes of code 3756

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel512(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x1f7
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 1

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 64
       mov      qword ptr [rsp+0x7C8], r15
       mov      qword ptr [rsp+0x7C0], r14
       mov      qword ptr [rsp+0x7B8], r13
       mov      qword ptr [rsp+0x7B0], r12
       mov      qword ptr [rsp+0x7A8], rbx
       lea      rbp, [rsp+0x40]
       mov      rcx, qword ptr [rbp+0x760]
       mov      rdi, qword ptr [rbp+0x750]
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      eax, dword ptr [rbp+0x7A8]
       mov      r8d, dword ptr [rbp+0x7B8]
       mov      r12d, dword ptr [rbp+0x73C]
       mov      r13d, dword ptr [rbp+0x728]
       vmovups  zmm0, zmmword ptr [rbp+0x6E0]
       vmovups  zmm6, zmmword ptr [rbp+0x6A0]
       vmovups  zmm1, zmmword ptr [rbp+0x660]
       vmovups  zmm7, zmmword ptr [rbp+0x620]
       vmovups  zmm2, zmmword ptr [rbp+0x5E0]
       vmovups  zmm8, zmmword ptr [rbp+0x5A0]
       vmovups  zmm3, zmmword ptr [rbp+0x560]
       vmovups  zmm9, zmmword ptr [rbp+0x520]
       vmovups  zmm4, zmmword ptr [rbp+0x4E0]
       vmovups  zmm10, zmmword ptr [rbp+0x4A0]
       vmovups  zmm5, zmmword ptr [rbp+0x460]
       vmovups  zmm11, zmmword ptr [rbp+0x420]
       mov      rbx, qword ptr [rbp+0x418]
       mov      r15, qword ptr [rbp+0x410]
       mov      r10d, dword ptr [rbp+0x40C]
       mov      r14d, dword ptr [rbp+0x408]
       mov      r11d, dword ptr [rbp+0x404]
 
G_M000_IG02:                ;; offset=0x0106
       jmp      G_M000_IG10
 
G_M000_IG03:                ;; offset=0x010B
       inc      r10d
 
G_M000_IG04:                ;; offset=0x010E
       cmp      r10d, edx
       jge      G_M000_IG12
 
G_M000_IG05:                ;; offset=0x0117
       xor      r9d, r9d
       mov      r14d, r9d
       jmp      SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x011F
       inc      r14d
       cmp      r14d, 3
       jge      SHORT G_M000_IG03
 
G_M000_IG07:                ;; offset=0x0128
       xor      r9d, r9d
       mov      dword ptr [rbp+0x74C], edx
       mov      qword ptr [rbp+0x750], rdi
       mov      dword ptr [rbp+0x748], esi
       mov      dword ptr [rbp+0x7B8], r8d
       mov      r11d, r9d
 
G_M000_IG08:                ;; offset=0x0148
       vmovups  zmm12, zmmword ptr [rbx]
       vmovups  zmm13, zmmword ptr [r15]
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 15
       add      r9d, r10d
       sar      r9d, 4
       mov      r8d, dword ptr [rbp+0x740]
       imul     r9d, r8d
       mov      esi, dword ptr [rbp+0x72C]
       mov      edi, esi
       imul     edi, eax
       add      edi, r9d
       add      edi, r14d
       imul     edi, r12d
       mov      r9d, r13d
       imul     r9d, eax
       add      edi, r9d
       add      edi, r11d
       shl      edi, 4
       movsxd   rdi, edi
       shl      rdi, 2
       add      rdi, rcx
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 15
       add      r9d, r10d
       and      r9d, -16
       mov      edx, r10d
       sub      edx, r9d
       movsxd   rdx, edx
       lea      rdx, [rdi+4*rdx]
       vbroadcastss zmm14, dword ptr [rdx]
       vfmadd231ps zmm0, zmm12, zmm14
       vfmadd231ps zmm6, zmm13, zmm14
       mov      edi, eax
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm1, zmm12, zmm14
       vfmadd231ps zmm7, zmm13, zmm14
       lea      edi, [rax+rax]
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm2, zmm12, zmm14
       vfmadd231ps zmm8, zmm13, zmm14
       imul     edi, eax, 48
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm3, zmm12, zmm14
       vfmadd231ps zmm9, zmm13, zmm14
       lea      edi, [4*rax]
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm4, zmm12, zmm14
       vfmadd231ps zmm10, zmm13, zmm14
       imul     edi, eax, 80
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm5, zmm12, zmm14
       vfmadd231ps zmm11, zmm13, zmm14
       add      rbx, 64
       add      r15, 64
       inc      r11d
       mov      dword ptr [rbp+0x72C], esi
       mov      dword ptr [rbp+0x740], r8d
 
G_M000_IG09:                ;; offset=0x0272
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      rdi, qword ptr [rbp+0x750]
       mov      r8d, dword ptr [rbp+0x7B8]
 
G_M000_IG10:                ;; offset=0x028C
       cmp      r11d, 3
       jge      G_M000_IG06
 
G_M000_IG11:                ;; offset=0x0296
       mov      dword ptr [rbp+0x74C], edx
       mov      qword ptr [rbp+0x750], rdi
       mov      dword ptr [rbp+0x748], esi
       mov      dword ptr [rbp+0x7B8], r8d
       jmp      G_M000_IG08
 
G_M000_IG12:                ;; offset=0x02B5
       mov      r10d, dword ptr [rbp+0x730]
       mov      r11d, r10d
       sar      r11d, 31
       and      r11d, 15
       add      r11d, r10d
       sar      r11d, 4
       mov      r14d, dword ptr [rbp+0x738]
       mov      ebx, r11d
       imul     ebx, r14d
       mov      r15d, dword ptr [rbp+0x72C]
       mov      dword ptr [rbp+0x7B8], r8d
       mov      r9d, r15d
       imul     r9d, r8d
       add      ebx, r9d
       add      ebx, r13d
       mov      r8d, ebx
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm0
       mov      dword ptr [rbp+0x730], r10d
       lea      r8d, [r10+0x10]
       cmp      r8d, esi
       jge      SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x0318
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm6
 
G_M000_IG14:                ;; offset=0x0334
       lea      r10d, [rbx+0x01]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm1
       cmp      r8d, esi
       jge      SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x034B
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       lea      r10d, [r10+r13+0x01]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm7
 
G_M000_IG16:                ;; offset=0x0369
       lea      r10d, [rbx+0x02]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm2
       cmp      r8d, esi
       jge      SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x0380
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       lea      r10d, [r10+r13+0x02]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm8
 
G_M000_IG18:                ;; offset=0x039E
       lea      r10d, [rbx+0x03]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm3
       cmp      r8d, esi
       jge      SHORT G_M000_IG20
 
G_M000_IG19:                ;; offset=0x03B5
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       lea      r10d, [r10+r13+0x03]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm9
 
G_M000_IG20:                ;; offset=0x03D3
       lea      r10d, [rbx+0x04]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm4
       cmp      r8d, esi
       jge      SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x03EA
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       lea      r10d, [r10+r13+0x04]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm10
 
G_M000_IG22:                ;; offset=0x0408
       add      ebx, 5
       shl      ebx, 4
       movsxd   r10, ebx
       vmovups  zmmword ptr [rdi+4*r10], zmm5
       mov      dword ptr [rbp+0x748], esi
       cmp      r8d, esi
       jge      SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x0423
       inc      r11d
       mov      dword ptr [rbp+0x738], r14d
       imul     r11d, r14d
       add      r9d, r11d
       lea      r8d, [r9+r13+0x05]
       shl      r8d, 4
       movsxd   r8, r8d
       mov      qword ptr [rbp+0x750], rdi
       vmovups  zmmword ptr [rdi+4*r8], zmm11
       mov      rdi, qword ptr [rbp+0x750]
       mov      r14d, dword ptr [rbp+0x738]
 
G_M000_IG24:                ;; offset=0x045C
       add      r13d, 6
 
G_M000_IG25:                ;; offset=0x0460
       lea      r8d, [r13+0x06]
       mov      r9d, dword ptr [rbp+0x7B8]
       cmp      r8d, r9d
       jg       G_M000_IG28
 
G_M000_IG26:                ;; offset=0x0474
       mov      r8d, r15d
       imul     r8d, r9d
       lea      r8d, [r8+r13+0x06]
       cmp      r8d, dword ptr [rbp+0x734]
       jg       G_M000_IG28
 
G_M000_IG27:                ;; offset=0x048D
       mov      dword ptr [rbp+0x7B8], r9d
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm7, ymm7, ymm7
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm8, ymm8, ymm8
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm9, ymm9, ymm9
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm10, ymm10, ymm10
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm11, ymm11, ymm11
       mov      ebx, dword ptr [rbp+0x730]
       mov      r8d, ebx
       imul     r8d, edx
       lea      r8d, [r8+8*r8]
       movsxd   r8, r8d
       mov      r10, qword ptr [rbp+0x758]
       lea      r8, [r10+4*r8]
       mov      dword ptr [rbp+0x74C], edx
       mov      r10d, edx
       shl      r10d, 4
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r8+4*r10]
       xor      r9d, r9d
       mov      dword ptr [rbp+0x72C], r15d
       mov      dword ptr [rbp+0x738], r14d
       mov      dword ptr [rbp+0x730], ebx
       mov      rbx, r8
       mov      r15, r10
       mov      r10d, r9d
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      r8d, dword ptr [rbp+0x7B8]
       jmp      G_M000_IG04
 
G_M000_IG28:                ;; offset=0x0537
       mov      r8d, r15d
       imul     r8d, eax
       mov      dword ptr [rbp-0x30], r8d
       jmp      G_M000_IG35
       align    [0 bytes for IG29]
 
G_M000_IG29:                ;; offset=0x0547
       vmulps   zmm7, zmm1, zmm7
       vaddps   zmm0, zmm7, zmm0
       vmulps   zmm1, zmm1, zmm2
       vaddps   zmm6, zmm1, zmm6
       jmp      G_M000_IG43
 
G_M000_IG30:                ;; offset=0x0564
       vmulps   zmm8, zmm1, zmm7
       vaddps   zmm0, zmm8, zmm0
       vmulps   zmm3, zmm1, zmm2
       vaddps   zmm6, zmm3, zmm6
       jmp      G_M000_IG45
 
G_M000_IG31:                ;; offset=0x0581
       mov      r11d, dword ptr [rbp+0x740]
 
G_M000_IG32:                ;; offset=0x0588
       mov      r10d, dword ptr [rbp+0x730]
       mov      esi, r10d
       sar      esi, 31
       and      esi, 15
       add      esi, r10d
       sar      esi, 4
       mov      r14d, dword ptr [rbp+0x738]
       mov      r8d, esi
       imul     r8d, r14d
       mov      r9d, dword ptr [rbp+0x7B8]
       mov      ebx, r15d
       imul     ebx, r9d
       add      r8d, ebx
       add      r8d, r13d
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm0
       lea      r8d, [r10+0x10]
       mov      ebx, dword ptr [rbp+0x748]
       cmp      r8d, ebx
       jge      SHORT G_M000_IG34
 
G_M000_IG33:                ;; offset=0x05DD
       inc      esi
       imul     esi, r14d
       mov      dword ptr [rbp+0x7B8], r9d
       mov      r8d, r15d
       imul     r8d, r9d
       add      esi, r8d
       add      esi, r13d
       shl      esi, 4
       movsxd   rsi, esi
       mov      qword ptr [rbp+0x750], rdi
       vmovups  zmmword ptr [rdi+4*rsi], zmm6
       mov      rdi, qword ptr [rbp+0x750]
       mov      r9d, dword ptr [rbp+0x7B8]
 
G_M000_IG34:                ;; offset=0x0619
       inc      r13d
       mov      dword ptr [rbp+0x740], r11d
       mov      dword ptr [rbp+0x748], ebx
       mov      dword ptr [rbp+0x730], r10d
 
G_M000_IG35:                ;; offset=0x0630
       cmp      r13d, r9d
       jge      G_M000_IG48
 
G_M000_IG36:                ;; offset=0x0639
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       mov      ebx, edx
       imul     ebx, dword ptr [rbp+0x730]
       lea      ebx, [rbx+8*rbx]
       movsxd   rbx, ebx
       mov      r10, qword ptr [rbp+0x758]
       lea      rbx, [r10+4*rbx]
       mov      r8d, edx
       shl      r8d, 4
       lea      r8d, [r8+8*r8]
       movsxd   r8, r8d
       lea      r8, [rbx+4*r8]
       mov      dword ptr [rbp+0x7B8], r9d
       mov      r10d, r15d
       imul     r10d, r9d
       add      r10d, r13d
       cmp      r10d, dword ptr [rbp+0x734]
       setl     r10b
       movzx    r10, r10b
       mov      dword ptr [rbp+0x2CC], r10d
       xor      r9d, r9d
       mov      r11d, r13d
       imul     r11d, eax
       mov      dword ptr [rbp-0x34], r11d
       cmp      r9d, edx
       mov      dword ptr [rbp+0x738], r14d
       jl       SHORT G_M000_IG39
       jmp      G_M000_IG31
 
G_M000_IG37:                ;; offset=0x06B3
       mov      rdi, qword ptr [rbp+0x750]
       inc      r9d
       cmp      r9d, edx
       jge      G_M000_IG32
 
G_M000_IG38:                ;; offset=0x06C6
       mov      dword ptr [rbp+0x740], r11d
 
G_M000_IG39:                ;; offset=0x06CD
       xor      r14d, r14d
       mov      esi, r9d
       sar      esi, 31
       and      esi, 15
       add      esi, r9d
       sar      esi, 4
       mov      r11d, dword ptr [rbp+0x740]
       imul     esi, r11d
       add      esi, dword ptr [rbp-0x30]
       mov      dword ptr [rbp-0x2C], esi
       mov      qword ptr [rbp+0x750], rdi
       jmp      G_M000_IG46
 
G_M000_IG40:                ;; offset=0x06FC
       vmulps   zmm7, zmm1, zmm7
       vaddps   zmm0, zmm7, zmm0
       vmulps   zmm1, zmm1, zmm2
       vaddps   zmm6, zmm1, zmm6
 
G_M000_IG41:                ;; offset=0x0714
       add      rbx, 64
       add      r8, 64
       lea      edi, [rsi+0x01]
       shl      edi, 4
       add      edi, r9d
       movsxd   rdi, edi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [rbx]
       vmovups  zmm2, zmmword ptr [r8]
       test     r10d, r10d
       je       G_M000_IG29
 
G_M000_IG42:                ;; offset=0x0744
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
 
G_M000_IG43:                ;; offset=0x0750
       add      rbx, 64
       add      r8, 64
       add      esi, 2
       shl      esi, 4
       add      esi, r9d
       movsxd   rdi, esi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [rbx]
       vmovups  zmm2, zmmword ptr [r8]
       test     r10d, r10d
       je       G_M000_IG30
 
G_M000_IG44:                ;; offset=0x0780
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
 
G_M000_IG45:                ;; offset=0x078C
       add      rbx, 64
       add      r8, 64
       inc      r14d
       cmp      r14d, 3
       mov      esi, dword ptr [rbp-0x2C]
       mov      r9d, dword ptr [rbp+0x2C8]
       jge      G_M000_IG37
 
G_M000_IG46:                ;; offset=0x07AB
       add      esi, r14d
       imul     esi, r12d
       add      esi, dword ptr [rbp-0x34]
       mov      edi, esi
       shl      edi, 4
       mov      r10d, r9d
       sar      r10d, 31
       and      r10d, 15
       add      r10d, r9d
       and      r10d, -16
       mov      dword ptr [rbp+0x2C8], r9d
       sub      r9d, r10d
       add      edi, r9d
       movsxd   rdi, edi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [rbx]
       vmovups  zmm2, zmmword ptr [r8]
       mov      r10d, dword ptr [rbp+0x2CC]
       test     r10d, r10d
       je       G_M000_IG40
 
G_M000_IG47:                ;; offset=0x07FF
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
       jmp      G_M000_IG41
 
G_M000_IG48:                ;; offset=0x0810
       inc      r15d
       mov      r8d, dword ptr [rbp+0x7B0]
       cmp      r15d, r8d
       jge      SHORT G_M000_IG52
 
G_M000_IG49:                ;; offset=0x081F
       xor      r13d, r13d
       mov      dword ptr [rbp+0x7B8], r9d
       mov      dword ptr [rbp+0x7B0], r8d
       jmp      G_M000_IG25
 
G_M000_IG50:                ;; offset=0x0835
       xor      r13d, r13d
       mov      dword ptr [rbp+0x7B0], r8d
       test     r8d, r8d
       mov      dword ptr [rbp+0x730], r15d
       mov      r8d, dword ptr [rbp+0x7B0]
       jle      SHORT G_M000_IG52
 
G_M000_IG51:                ;; offset=0x0852
       mov      r15d, r13d
       jmp      SHORT G_M000_IG49
 
G_M000_IG52:                ;; offset=0x0857
       mov      r15d, dword ptr [rbp+0x730]
       add      r15d, 32
       cmp      r15d, dword ptr [rbp+0x748]
       jl       SHORT G_M000_IG50
 
G_M000_IG53:                ;; offset=0x086B
       vzeroupper 
       add      rsp, 0x7A8
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 2176

; Assembly listing for method KernelAccess:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier0 code
; rbp based frame
; partially interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 64
       lea      rbp, [rsp+0x40]
       xor      eax, eax
       mov      qword ptr [rbp-0x28], rax
       mov      dword ptr [rbp-0x04], edi
       mov      dword ptr [rbp-0x08], esi
       mov      dword ptr [rbp-0x0C], edx
       mov      dword ptr [rbp-0x10], ecx
       mov      bword ptr [rbp-0x18], r8
       mov      bword ptr [rbp-0x20], r9
 
G_M000_IG02:                ;; offset=0x0024
       mov      rax, bword ptr [rbp-0x20]
       mov      bword ptr [rsp], rax
       mov      rax, bword ptr [rbp+0x10]
       mov      bword ptr [rsp+0x08], rax
       mov      rax, 0x7A7CAD800198
       mov      rax, gword ptr [rax]
       mov      gword ptr [rbp-0x28], rax
       mov      rax, gword ptr [rbp-0x28]
       mov      esi, dword ptr [rbp-0x04]
       mov      edx, dword ptr [rbp-0x08]
       mov      ecx, dword ptr [rbp-0x0C]
       mov      r8d, dword ptr [rbp-0x10]
       mov      r9, bword ptr [rbp-0x18]
       mov      rdi, gword ptr [rax+0x08]
       mov      rax, gword ptr [rbp-0x28]
       call     [rax+0x18]KernelAccess+PlanCall:Invoke(int,int,int,int,byref,byref,byref):bool:this
       nop      
 
G_M000_IG03:                ;; offset=0x0067
       add      rsp, 64
       pop      rbp
       ret      
 
; Total bytes of code 109

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 80
       lea      rbp, [rsp+0x50]
       vxorps   xmm8, xmm8, xmm8
       vmovdqu  ymmword ptr [rbp-0x50], ymm8
       vmovdqa  xmmword ptr [rbp-0x30], xmm8
       mov      dword ptr [rbp-0x04], edi
       mov      dword ptr [rbp-0x08], esi
       mov      dword ptr [rbp-0x0C], edx
       mov      dword ptr [rbp-0x10], ecx
       mov      bword ptr [rbp-0x18], r8
       mov      bword ptr [rbp-0x20], r9
 
G_M000_IG02:                ;; offset=0x002D
       xor      eax, eax
       mov      dword ptr [rbp-0x24], eax
       mov      rax, bword ptr [rbp+0x10]
       xor      ecx, ecx
       mov      dword ptr [rax], ecx
       mov      rax, bword ptr [rbp-0x20]
       mov      ecx, dword ptr [rbp-0x24]
       mov      dword ptr [rax], ecx
       mov      rax, bword ptr [rbp-0x18]
       mov      ecx, dword ptr [rbp-0x24]
       mov      dword ptr [rax], ecx
       cmp      dword ptr [rbp-0x04], 0
       jle      SHORT G_M000_IG03
       cmp      dword ptr [rbp-0x08], 0
       jle      SHORT G_M000_IG03
       cmp      dword ptr [rbp-0x0C], 0
       jle      SHORT G_M000_IG03
       cmp      dword ptr [rbp-0x10], 0
       jg       SHORT G_M000_IG05
 
G_M000_IG03:                ;; offset=0x0064
       xor      eax, eax
 
G_M000_IG04:                ;; offset=0x0066
       add      rsp, 80
       pop      rbp
       ret      
 
G_M000_IG05:                ;; offset=0x006C
       movsxd   rax, dword ptr [rbp-0x04]
       imul     rax, rax, 16
       jo       SHORT G_M000_IG06
       imul     rax, rax, 8
       jo       SHORT G_M000_IG06
       mov      qword ptr [rbp-0x30], rax
       movsxd   rax, dword ptr [rbp-0x08]
       imul     rax, rax, 16
       jo       SHORT G_M000_IG06
       imul     rax, rax, 8
       jo       SHORT G_M000_IG06
       mov      qword ptr [rbp-0x38], rax
       movsxd   rax, dword ptr [rbp-0x08]
       movsxd   rcx, dword ptr [rbp-0x0C]
       imul     rax, rcx
       jo       SHORT G_M000_IG06
       movsxd   rcx, dword ptr [rbp-0x10]
       imul     rax, rcx
       jo       SHORT G_M000_IG06
       mov      qword ptr [rbp-0x40], rax
       mov      rax, qword ptr [rbp-0x30]
       add      rax, qword ptr [rbp-0x38]
       jo       SHORT G_M000_IG06
       add      rax, qword ptr [rbp-0x40]
       jo       SHORT G_M000_IG06
       cmp      rax, 0x1000000
       jle      SHORT G_M000_IG07
       xor      eax, eax
       mov      dword ptr [rbp-0x44], eax
       jmp      SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x00CF
       call     CORINFO_HELP_OVERFLOW
       int3     
 
G_M000_IG07:                ;; offset=0x00D5
       mov      rax, bword ptr [rbp-0x18]
       mov      ecx, dword ptr [rbp-0x30]
       mov      dword ptr [rax], ecx
       mov      rax, bword ptr [rbp-0x20]
       mov      ecx, dword ptr [rbp-0x38]
       mov      dword ptr [rax], ecx
       mov      rax, bword ptr [rbp+0x10]
       mov      ecx, dword ptr [rbp-0x40]
       mov      dword ptr [rax], ecx
       mov      dword ptr [rbp-0x44], 1
 
G_M000_IG08:                ;; offset=0x00F7
       mov      eax, dword ptr [rbp-0x44]
 
G_M000_IG09:                ;; offset=0x00FA
       add      rsp, 80
       pop      rbp
       ret      
 
G_M000_IG10:                ;; offset=0x0100
       push     rax
 
G_M000_IG11:                ;; offset=0x0101
       mov      gword ptr [rbp-0x50], rdi
       xor      eax, eax
       mov      dword ptr [rbp-0x44], eax
       lea      rax, G_M000_IG08
 
G_M000_IG12:                ;; offset=0x0111
       add      rsp, 8
       ret      
 
; Total bytes of code 278

; Assembly listing for method KernelAccess:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier0 code
; rbp based frame
; partially interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 192
       lea      rbp, [rsp+0xC0]
       xor      eax, eax
       mov      qword ptr [rbp-0x38], rax
       mov      bword ptr [rbp-0x10], rdi
       mov      qword ptr [rbp-0x08], rsi
       mov      bword ptr [rbp-0x20], rdx
       mov      qword ptr [rbp-0x18], rcx
       mov      bword ptr [rbp-0x30], r8
       mov      qword ptr [rbp-0x28], r9
 
G_M000_IG02:                ;; offset=0x002E
       lea      rdi, [rsp]
       lea      rsi, [rbp-0x30]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       lea      rdi, [rsp+0x10]
       lea      rsi, [rbp+0x10]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp+0x10], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       lea      rdi, [rsp+0x20]
       lea      rsi, [rbp+0x20]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp+0x20], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       lea      rdi, [rsp+0x30]
       lea      rsi, [rbp+0x30]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp+0x30], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       lea      rdi, [rsp+0x40]
       lea      rsi, [rbp+0x40]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp+0x40], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       lea      rdi, [rsp+0x50]
       lea      rsi, [rbp+0x50]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp+0x50], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       mov      eax, dword ptr [rbp+0x68]
       mov      dword ptr [rsp+0x60], eax
       mov      eax, dword ptr [rbp+0x70]
       mov      dword ptr [rsp+0x68], eax
       mov      eax, dword ptr [rbp+0x78]
       mov      dword ptr [rsp+0x70], eax
       mov      eax, dword ptr [rbp+0x80]
       mov      dword ptr [rsp+0x78], eax
       movzx    rax, byte  ptr [rbp+0x88]
       mov      dword ptr [rsp+0x80], eax
       mov      rax, 0x7A7CAD8001A8
       mov      rax, gword ptr [rax]
       mov      gword ptr [rbp-0x38], rax
       mov      rax, gword ptr [rbp-0x38]
       mov      rsi, bword ptr [rbp-0x10]
       mov      rdx, qword ptr [rbp-0x08]
       mov      rcx, bword ptr [rbp-0x20]
       mov      r8, qword ptr [rbp-0x18]
       mov      r9d, dword ptr [rbp+0x60]
       mov      rdi, gword ptr [rax+0x08]
 
G_M000_IG03:                ;; offset=0x0128
       mov      rax, gword ptr [rbp-0x38]
       call     [rax+0x18]KernelAccess+WinogradCall:Invoke(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool:this
       nop      
 
G_M000_IG04:                ;; offset=0x0130
       add      rsp, 192
       pop      rbp
       ret      
 
; Total bytes of code 313

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; partially interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x480
       lea      rbp, [rsp+0x480]
       xor      eax, eax
       mov      qword ptr [rbp-0x428], rax
       vxorps   xmm8, xmm8, xmm8
       mov      rax, -960
       vmovdqa  xmmword ptr [rbp+rax-0x60], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x50], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x60], rax
       mov      bword ptr [rbp-0x38], rdi
       mov      qword ptr [rbp-0x30], rsi
       mov      bword ptr [rbp-0x48], rdx
       mov      qword ptr [rbp-0x40], rcx
       mov      bword ptr [rbp-0x58], r8
       mov      qword ptr [rbp-0x50], r9
 
G_M000_IG02:                ;; offset=0x005C
       mov      dword ptr [rbp-0x418], 0x3E8
       mov      edi, dword ptr [rbp+0x60]
       mov      esi, dword ptr [rbp+0x68]
       mov      edx, dword ptr [rbp+0x70]
       mov      ecx, dword ptr [rbp+0x78]
       mov      r9d, dword ptr [rbp+0x80]
       mov      r8d, 1
       call     [Lokad.Onnx.ConvBlockedSpatial:Geometry(int,int,int,int,int,int)]
       lea      rax, [rbp-0x70]
       mov      qword ptr [rsp], rax
       lea      r9, [rbp-0x68]
       lea      r8, [rbp-0x60]
       mov      edi, dword ptr [rbp+0x60]
       mov      esi, dword ptr [rbp+0x68]
       mov      edx, dword ptr [rbp+0x70]
       mov      ecx, dword ptr [rbp+0x78]
       call     [Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool]
       test     eax, eax
       jne      SHORT G_M000_IG03
       mov      rdi, 0x7A8492761790
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG96
 
G_M000_IG03:                ;; offset=0x00BF
       mov      eax, dword ptr [rbp+0x60]
       imul     eax, dword ptr [rbp+0x70]
       jo       G_M000_IG102
       imul     eax, dword ptr [rbp+0x78]
       jo       G_M000_IG102
       cmp      dword ptr [rbp-0x30], eax
       jne      G_M000_IG07
       imul     eax, dword ptr [rbp+0x60], 16
       jo       G_M000_IG102
       imul     eax, dword ptr [rbp+0x68]
       jo       G_M000_IG102
       cmp      dword ptr [rbp-0x40], eax
       jne      G_M000_IG13
       mov      eax, dword ptr [rbp+0x28]
       cmp      eax, dword ptr [rbp-0x70]
       jne      G_M000_IG12
       cmp      dword ptr [rbp-0x50], 0
       je       SHORT G_M000_IG04
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp+0x68]
       jne      G_M000_IG11
       mov      rdi, 0x7A8492761794
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG04:                ;; offset=0x0129
       cmp      dword ptr [rbp+0x18], 0
       je       SHORT G_M000_IG05
       mov      eax, dword ptr [rbp+0x18]
       cmp      eax, dword ptr [rbp-0x70]
       jne      G_M000_IG10
       mov      rdi, 0x7A8492761798
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG05:                ;; offset=0x014A
       mov      eax, dword ptr [rbp+0x38]
       cmp      eax, dword ptr [rbp-0x60]
       jl       G_M000_IG09
       mov      eax, dword ptr [rbp+0x48]
       cmp      eax, dword ptr [rbp-0x68]
       jl       SHORT G_M000_IG08
       mov      eax, dword ptr [rbp+0x58]
       cmp      eax, dword ptr [rbp-0x70]
       jge      G_M000_IG14
 
G_M000_IG06:                ;; offset=0x016A
       mov      rdi, 0x7A849276179C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG07:                ;; offset=0x0179
       mov      rdi, 0x7A8491C8CD28
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0xC8], rax
       mov      edi, 0x102B4
       mov      rsi, 0x7A84924B9F30
       call     [CORINFO_HELP_STRCNS]
       mov      gword ptr [rbp-0x420], rax
       mov      rsi, gword ptr [rbp-0x420]
       mov      rdi, gword ptr [rbp-0xC8]
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, gword ptr [rbp-0xC8]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG08:                ;; offset=0x01CC
       mov      rdi, 0x7A84927617A0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG07
 
G_M000_IG09:                ;; offset=0x01DD
       mov      rdi, 0x7A84927617A4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG07
 
G_M000_IG10:                ;; offset=0x01EE
       mov      rdi, 0x7A84927617A8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG11:                ;; offset=0x0202
       mov      rdi, 0x7A84927617AC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG12:                ;; offset=0x0216
       mov      rdi, 0x7A84927617B0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG13:                ;; offset=0x022A
       mov      rdi, 0x7A84927617B4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG14:                ;; offset=0x023E
       lea      rdi, [rbp+0x30]
       mov      edx, dword ptr [rbp-0x60]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xD8], rax
       mov      qword ptr [rbp-0xD0], rdx
 
G_M000_IG15:                ;; offset=0x025B
       vmovdqu  xmm0, xmmword ptr [rbp-0xD8]
       vmovdqu  xmmword ptr [rbp+0x30], xmm0
 
G_M000_IG16:                ;; offset=0x0268
       lea      rdi, [rbp+0x40]
       mov      edx, dword ptr [rbp-0x68]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xE8], rax
       mov      qword ptr [rbp-0xE0], rdx
 
G_M000_IG17:                ;; offset=0x0285
       vmovdqu  xmm0, xmmword ptr [rbp-0xE8]
       vmovdqu  xmmword ptr [rbp+0x40], xmm0
 
G_M000_IG18:                ;; offset=0x0292
       lea      rdi, [rbp+0x50]
       mov      edx, dword ptr [rbp-0x70]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xF8], rax
       mov      qword ptr [rbp-0xF0], rdx
 
G_M000_IG19:                ;; offset=0x02AF
       vmovdqu  xmm0, xmmword ptr [rbp-0xF8]
       vmovdqu  xmmword ptr [rbp+0x50], xmm0
 
G_M000_IG20:                ;; offset=0x02BC
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu  xmmword ptr [rbp-0x118], xmm0
 
G_M000_IG21:                ;; offset=0x02C9
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x108], rax
       mov      qword ptr [rbp-0x100], rdx
       mov      rdx, bword ptr [rbp-0x108]
       mov      rcx, qword ptr [rbp-0x100]
       mov      rdi, bword ptr [rbp-0x118]
       mov      rsi, qword ptr [rbp-0x110]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG55
 
G_M000_IG22:                ;; offset=0x030F
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu32 xmmword ptr [rbp-0x140], xmm0
 
G_M000_IG23:                ;; offset=0x031B
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x130], rax
       mov      qword ptr [rbp-0x128], rdx
       mov      rdx, bword ptr [rbp-0x130]
       mov      rcx, qword ptr [rbp-0x128]
       mov      rdi, bword ptr [rbp-0x140]
       mov      rsi, qword ptr [rbp-0x138]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG75
 
G_M000_IG24:                ;; offset=0x0361
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu32 xmmword ptr [rbp-0x160], xmm0
 
G_M000_IG25:                ;; offset=0x036D
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x150], rax
       mov      qword ptr [rbp-0x148], rdx
       mov      rdx, bword ptr [rbp-0x150]
       mov      rcx, qword ptr [rbp-0x148]
       mov      rdi, bword ptr [rbp-0x160]
       mov      rsi, qword ptr [rbp-0x158]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG74
 
G_M000_IG26:                ;; offset=0x03B3
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu32 xmmword ptr [rbp-0x180], xmm0
 
G_M000_IG27:                ;; offset=0x03BF
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x170], rax
       mov      qword ptr [rbp-0x168], rdx
       mov      rdx, bword ptr [rbp-0x170]
       mov      rcx, qword ptr [rbp-0x168]
       mov      rdi, bword ptr [rbp-0x180]
       mov      rsi, qword ptr [rbp-0x178]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG73
 
G_M000_IG28:                ;; offset=0x0405
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu32 xmmword ptr [rbp-0x1A0], xmm0
 
G_M000_IG29:                ;; offset=0x0411
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x190], rax
       mov      qword ptr [rbp-0x188], rdx
       mov      rdx, bword ptr [rbp-0x190]
       mov      rcx, qword ptr [rbp-0x188]
       mov      rdi, bword ptr [rbp-0x1A0]
       mov      rsi, qword ptr [rbp-0x198]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG72
 
G_M000_IG30:                ;; offset=0x0457
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu32 xmmword ptr [rbp-0x1C0], xmm0
 
G_M000_IG31:                ;; offset=0x0463
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x1B0], rax
       mov      qword ptr [rbp-0x1A8], rdx
       mov      rdx, bword ptr [rbp-0x1B0]
       mov      rcx, qword ptr [rbp-0x1A8]
       mov      rdi, bword ptr [rbp-0x1C0]
       mov      rsi, qword ptr [rbp-0x1B8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG71
 
G_M000_IG32:                ;; offset=0x04A9
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu32 xmmword ptr [rbp-0x1E0], xmm0
 
G_M000_IG33:                ;; offset=0x04B5
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x1D0], rax
       mov      qword ptr [rbp-0x1C8], rdx
       mov      rdx, bword ptr [rbp-0x1D0]
       mov      rcx, qword ptr [rbp-0x1C8]
       mov      rdi, bword ptr [rbp-0x1E0]
       mov      rsi, qword ptr [rbp-0x1D8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG70
 
G_M000_IG34:                ;; offset=0x04FB
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu32 xmmword ptr [rbp-0x200], xmm0
 
G_M000_IG35:                ;; offset=0x0507
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x1F0], rax
       mov      qword ptr [rbp-0x1E8], rdx
       mov      rdx, bword ptr [rbp-0x1F0]
       mov      rcx, qword ptr [rbp-0x1E8]
       mov      rdi, bword ptr [rbp-0x200]
       mov      rsi, qword ptr [rbp-0x1F8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG69
 
G_M000_IG36:                ;; offset=0x054D
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu32 xmmword ptr [rbp-0x220], xmm0
 
G_M000_IG37:                ;; offset=0x0559
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x210], rax
       mov      qword ptr [rbp-0x208], rdx
       mov      rdx, bword ptr [rbp-0x210]
       mov      rcx, qword ptr [rbp-0x208]
       mov      rdi, bword ptr [rbp-0x220]
       mov      rsi, qword ptr [rbp-0x218]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG68
 
G_M000_IG38:                ;; offset=0x059F
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu32 xmmword ptr [rbp-0x240], xmm0
 
G_M000_IG39:                ;; offset=0x05AB
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x230], rax
       mov      qword ptr [rbp-0x228], rdx
       mov      rdx, bword ptr [rbp-0x230]
       mov      rcx, qword ptr [rbp-0x228]
       mov      rdi, bword ptr [rbp-0x240]
       mov      rsi, qword ptr [rbp-0x238]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG67
 
G_M000_IG40:                ;; offset=0x05F1
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu32 xmmword ptr [rbp-0x260], xmm0
 
G_M000_IG41:                ;; offset=0x05FD
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x250], rax
       mov      qword ptr [rbp-0x248], rdx
       mov      rdx, bword ptr [rbp-0x250]
       mov      rcx, qword ptr [rbp-0x248]
       mov      rdi, bword ptr [rbp-0x260]
       mov      rsi, qword ptr [rbp-0x258]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG66
 
G_M000_IG42:                ;; offset=0x0643
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu32 xmmword ptr [rbp-0x280], xmm0
 
G_M000_IG43:                ;; offset=0x064F
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x270], rax
       mov      qword ptr [rbp-0x268], rdx
       mov      rdx, bword ptr [rbp-0x270]
       mov      rcx, qword ptr [rbp-0x268]
       mov      rdi, bword ptr [rbp-0x280]
       mov      rsi, qword ptr [rbp-0x278]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG65
 
G_M000_IG44:                ;; offset=0x0695
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu32 xmmword ptr [rbp-0x2A0], xmm0
 
G_M000_IG45:                ;; offset=0x06A1
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x290], rax
       mov      qword ptr [rbp-0x288], rdx
       mov      rdx, bword ptr [rbp-0x290]
       mov      rcx, qword ptr [rbp-0x288]
       mov      rdi, bword ptr [rbp-0x2A0]
       mov      rsi, qword ptr [rbp-0x298]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG64
 
G_M000_IG46:                ;; offset=0x06E7
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu32 xmmword ptr [rbp-0x2C0], xmm0
 
G_M000_IG47:                ;; offset=0x06F3
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x2B0], rax
       mov      qword ptr [rbp-0x2A8], rdx
       mov      rdx, bword ptr [rbp-0x2B0]
       mov      rcx, qword ptr [rbp-0x2A8]
       mov      rdi, bword ptr [rbp-0x2C0]
       mov      rsi, qword ptr [rbp-0x2B8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG63
 
G_M000_IG48:                ;; offset=0x0739
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu32 xmmword ptr [rbp-0x2E0], xmm0
 
G_M000_IG49:                ;; offset=0x0745
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x2D0], rax
       mov      qword ptr [rbp-0x2C8], rdx
       mov      rdx, bword ptr [rbp-0x2D0]
       mov      rcx, qword ptr [rbp-0x2C8]
       mov      rdi, bword ptr [rbp-0x2E0]
       mov      rsi, qword ptr [rbp-0x2D8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG62
 
G_M000_IG50:                ;; offset=0x078B
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu32 xmmword ptr [rbp-0x300], xmm0
 
G_M000_IG51:                ;; offset=0x0797
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x2F0], rax
       mov      qword ptr [rbp-0x2E8], rdx
       mov      rdx, bword ptr [rbp-0x2F0]
       mov      rcx, qword ptr [rbp-0x2E8]
       mov      rdi, bword ptr [rbp-0x300]
       mov      rsi, qword ptr [rbp-0x2F8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG61
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x310], rax
       mov      qword ptr [rbp-0x308], rdx
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x320], rax
       mov      qword ptr [rbp-0x318], rdx
       mov      rdx, bword ptr [rbp-0x320]
       mov      rcx, qword ptr [rbp-0x318]
       mov      rdi, bword ptr [rbp-0x310]
       mov      rsi, qword ptr [rbp-0x308]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG60
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x330], rax
       mov      qword ptr [rbp-0x328], rdx
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x340], rax
       mov      qword ptr [rbp-0x338], rdx
       mov      rdx, bword ptr [rbp-0x340]
       mov      rcx, qword ptr [rbp-0x338]
       mov      rdi, bword ptr [rbp-0x330]
       mov      rsi, qword ptr [rbp-0x328]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG59
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x350], rax
       mov      qword ptr [rbp-0x348], rdx
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x360], rax
       mov      qword ptr [rbp-0x358], rdx
 
G_M000_IG52:                ;; offset=0x08D9
       mov      rdx, bword ptr [rbp-0x360]
       mov      rcx, qword ptr [rbp-0x358]
       mov      rdi, bword ptr [rbp-0x350]
       mov      rsi, qword ptr [rbp-0x348]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG58
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x370], rax
       mov      qword ptr [rbp-0x368], rdx
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x380], rax
       mov      qword ptr [rbp-0x378], rdx
       mov      rdx, bword ptr [rbp-0x380]
       mov      rcx, qword ptr [rbp-0x378]
       mov      rdi, bword ptr [rbp-0x370]
       mov      rsi, qword ptr [rbp-0x368]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG57
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x390], rax
       mov      qword ptr [rbp-0x388], rdx
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3A0], rax
       mov      qword ptr [rbp-0x398], rdx
       mov      rdx, bword ptr [rbp-0x3A0]
       mov      rcx, qword ptr [rbp-0x398]
       mov      rdi, bword ptr [rbp-0x390]
       mov      rsi, qword ptr [rbp-0x388]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG56
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3B0], rax
       mov      qword ptr [rbp-0x3A8], rdx
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3C0], rax
       mov      qword ptr [rbp-0x3B8], rdx
       mov      rdx, bword ptr [rbp-0x3C0]
       mov      rcx, qword ptr [rbp-0x3B8]
       mov      rdi, bword ptr [rbp-0x3B0]
       mov      rsi, qword ptr [rbp-0x3A8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
 
G_M000_IG53:                ;; offset=0x0A21
       test     eax, eax
       je       G_M000_IG76
 
G_M000_IG54:                ;; offset=0x0A29
       mov      rdi, 0x7A84927617B8
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG55:                ;; offset=0x0A38
       mov      rdi, 0x7A8491C8CD28
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0x120], rax
       mov      edi, 0x102E2
       mov      rsi, 0x7A84924B9F30
       call     [CORINFO_HELP_STRCNS]
       mov      gword ptr [rbp-0x428], rax
       mov      rsi, gword ptr [rbp-0x428]
       mov      rdi, gword ptr [rbp-0x120]
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, gword ptr [rbp-0x120]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG56:                ;; offset=0x0A8B
       mov      rdi, 0x7A84927617BC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG55
 
G_M000_IG57:                ;; offset=0x0A9C
       mov      rdi, 0x7A84927617C0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG55
 
G_M000_IG58:                ;; offset=0x0AAD
       mov      rdi, 0x7A84927617C4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG59:                ;; offset=0x0AC1
       mov      rdi, 0x7A84927617C8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG60:                ;; offset=0x0AD5
       mov      rdi, 0x7A84927617CC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG61:                ;; offset=0x0AE9
       mov      rdi, 0x7A84927617D0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG62:                ;; offset=0x0AFD
       mov      rdi, 0x7A84927617D4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG63:                ;; offset=0x0B11
       mov      rdi, 0x7A84927617D8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG64:                ;; offset=0x0B25
       mov      rdi, 0x7A84927617DC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG65:                ;; offset=0x0B39
       mov      rdi, 0x7A84927617E0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG66:                ;; offset=0x0B4D
       mov      rdi, 0x7A84927617E4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG67:                ;; offset=0x0B61
       mov      rdi, 0x7A84927617E8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG68:                ;; offset=0x0B75
       mov      rdi, 0x7A84927617EC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG69:                ;; offset=0x0B89
       mov      rdi, 0x7A84927617F0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG70:                ;; offset=0x0B9D
       mov      rdi, 0x7A84927617F4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG71:                ;; offset=0x0BB1
       mov      rdi, 0x7A84927617F8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG72:                ;; offset=0x0BC5
       mov      rdi, 0x7A84927617FC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG73:                ;; offset=0x0BD9
       mov      rdi, 0x7A8492761800
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG74:                ;; offset=0x0BED
       mov      rdi, 0x7A8492761804
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG75:                ;; offset=0x0C01
       mov      rdi, 0x7A8492761808
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG76:                ;; offset=0x0C15
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG77
       mov      rdi, 0x7A849276180C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG77:                ;; offset=0x0C2D
       cmp      dword ptr [rbp+0x80], 8
       jne      SHORT G_M000_IG78
       mov      rdi, 0x7A8492761810
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG78:                ;; offset=0x0C45
       mov      rdi, bword ptr [rbp-0x38]
       mov      rsi, qword ptr [rbp-0x30]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG80
       mov      rdi, bword ptr [rbp-0x48]
       mov      rsi, qword ptr [rbp-0x40]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG82
       mov      rdi, bword ptr [rbp-0x58]
       mov      rsi, qword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG81
       mov      rdi, bword ptr [rbp+0x10]
       mov      rsi, qword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG83
 
G_M000_IG79:                ;; offset=0x0C8D
       mov      rdi, 0x7A8492761814
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG80:                ;; offset=0x0C9C
       mov      rdi, 0x7A8492761818
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG96
 
G_M000_IG81:                ;; offset=0x0CB0
       mov      rdi, 0x7A849276181C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG80
 
G_M000_IG82:                ;; offset=0x0CC1
       mov      rdi, 0x7A8492761820
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG80
 
G_M000_IG83:                ;; offset=0x0CD2
       mov      eax, dword ptr [rbp+0x78]
       inc      eax
       mov      dword ptr [rbp-0x44C], eax
       mov      eax, dword ptr [rbp-0x44C]
       shr      eax, 31
       add      eax, dword ptr [rbp-0x44C]
       sar      eax, 1
       mov      dword ptr [rbp-0x74], eax
       mov      eax, dword ptr [rbp+0x70]
       add      eax, 1
       jo       G_M000_IG102
       mov      dword ptr [rbp-0x450], eax
       mov      eax, dword ptr [rbp-0x450]
       shr      eax, 31
       add      eax, dword ptr [rbp-0x450]
       sar      eax, 1
       imul     eax, dword ptr [rbp-0x74]
       jo       G_M000_IG102
       mov      dword ptr [rbp-0x78], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x7C], eax
       jmp      G_M000_IG91
 
G_M000_IG84:                ;; offset=0x0D2B
       mov      eax, dword ptr [rbp-0x78]
       mov      esi, eax
       sub      esi, dword ptr [rbp-0x7C]
       mov      edi, 8
       call     [System.Math:Min(int,int):int]
       mov      dword ptr [rbp-0x80], eax
       mov      eax, dword ptr [rbp+0x78]
       mov      dword ptr [rsp], eax
       mov      eax, dword ptr [rbp-0x74]
       mov      dword ptr [rsp+0x08], eax
       mov      eax, dword ptr [rbp-0x7C]
       mov      dword ptr [rsp+0x10], eax
       mov      eax, dword ptr [rbp-0x80]
       mov      dword ptr [rsp+0x18], eax
       mov      rdx, bword ptr [rbp+0x30]
       mov      rcx, qword ptr [rbp+0x38]
       mov      rdi, bword ptr [rbp-0x38]
       mov      rsi, qword ptr [rbp-0x30]
       mov      r8d, dword ptr [rbp+0x60]
       mov      r9d, dword ptr [rbp+0x70]
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3D0], rax
       mov      qword ptr [rbp-0x3C8], rdx
       mov      rdi, bword ptr [rbp-0x3D0]
       mov      rsi, qword ptr [rbp-0x3C8]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG85
       mov      rdi, 0x7A8492761824
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG96
 
G_M000_IG85:                ;; offset=0x0DC2
       lea      rdi, [rbp+0x30]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xA8], rax
       mov      rax, bword ptr [rbp-0xA8]
       mov      qword ptr [rbp-0x430], rax
       mov      rax, qword ptr [rbp-0x430]
       mov      qword ptr [rbp-0x88], rax
       lea      rdi, [rbp-0x48]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xB0], rax
       mov      rax, bword ptr [rbp-0xB0]
       mov      qword ptr [rbp-0x438], rax
       mov      rax, qword ptr [rbp-0x438]
       mov      qword ptr [rbp-0x90], rax
       lea      rdi, [rbp+0x40]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xB8], rax
       mov      rax, bword ptr [rbp-0xB8]
       mov      qword ptr [rbp-0x440], rax
       mov      rax, qword ptr [rbp-0x440]
       mov      qword ptr [rbp-0x98], rax
       lea      rdi, [rbp+0x50]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xC0], rax
       mov      rax, bword ptr [rbp-0xC0]
       mov      qword ptr [rbp-0x448], rax
       mov      rax, qword ptr [rbp-0x448]
       mov      qword ptr [rbp-0xA0], rax
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG86
       mov      rdi, qword ptr [rbp-0x88]
       mov      rsi, qword ptr [rbp-0x90]
       mov      rdx, qword ptr [rbp-0x98]
       mov      ecx, dword ptr [rbp+0x60]
       mov      r8d, dword ptr [rbp+0x68]
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int)]
       jmp      SHORT G_M000_IG87
 
G_M000_IG86:                ;; offset=0x0EA3
       mov      rdi, 0x7A8492761828
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rdi, qword ptr [rbp-0x88]
       mov      rsi, qword ptr [rbp-0x90]
       mov      rdx, qword ptr [rbp-0x98]
       mov      ecx, dword ptr [rbp+0x60]
       mov      r8d, dword ptr [rbp+0x68]
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int)]
 
G_M000_IG87:                ;; offset=0x0ED4
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3E0], rax
       mov      qword ptr [rbp-0x3D8], rdx
       mov      rdi, bword ptr [rbp-0x3E0]
       mov      rsi, qword ptr [rbp-0x3D8]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG88
       mov      rdi, 0x7A849276182C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG96
 
G_M000_IG88:                ;; offset=0x0F1C
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG89
       mov      eax, dword ptr [rbp-0x7C]
       mov      dword ptr [rsp], eax
       mov      eax, dword ptr [rbp-0x80]
       mov      dword ptr [rsp+0x08], eax
       mov      rdi, qword ptr [rbp-0x98]
       mov      rsi, qword ptr [rbp-0xA0]
       mov      edx, dword ptr [rbp+0x68]
       mov      ecx, dword ptr [rbp+0x70]
       mov      r8d, dword ptr [rbp+0x78]
       mov      r9d, dword ptr [rbp-0x74]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd512(ptr,ptr,int,int,int,int,int,int)]
       jmp      SHORT G_M000_IG90
 
G_M000_IG89:                ;; offset=0x0F56
       mov      rdi, 0x7A8492761830
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x7C]
       mov      dword ptr [rsp], eax
       mov      eax, dword ptr [rbp-0x80]
       mov      dword ptr [rsp+0x08], eax
       mov      rdi, qword ptr [rbp-0x98]
       mov      rsi, qword ptr [rbp-0xA0]
       mov      edx, dword ptr [rbp+0x68]
       mov      ecx, dword ptr [rbp+0x70]
       mov      r8d, dword ptr [rbp+0x78]
       mov      r9d, dword ptr [rbp-0x74]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
 
G_M000_IG90:                ;; offset=0x0F94
       mov      rdi, 0x7A8492761834
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
       mov      bword ptr [rbp-0xA8], rax
       xor      eax, eax
       mov      bword ptr [rbp-0xB0], rax
       xor      eax, eax
       mov      bword ptr [rbp-0xB8], rax
       xor      eax, eax
       mov      bword ptr [rbp-0xC0], rax
       mov      eax, dword ptr [rbp-0x7C]
       add      eax, 8
       mov      dword ptr [rbp-0x7C], eax
 
G_M000_IG91:                ;; offset=0x0FD0
       mov      eax, dword ptr [rbp-0x418]
       dec      eax
       mov      dword ptr [rbp-0x418], eax
       cmp      dword ptr [rbp-0x418], 0
       jg       SHORT G_M000_IG93
 
G_M000_IG92:                ;; offset=0x0FE7
       lea      rdi, [rbp-0x418]
       mov      esi, 943
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG93:                ;; offset=0x0FF8
       mov      eax, dword ptr [rbp-0x7C]
       cmp      eax, dword ptr [rbp-0x78]
       jl       G_M000_IG84
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3F0], rax
       mov      qword ptr [rbp-0x3E8], rdx
       mov      rdi, bword ptr [rbp-0x3F0]
       mov      rsi, qword ptr [rbp-0x3E8]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG95
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x400], rax
       mov      qword ptr [rbp-0x3F8], rdx
       mov      rdi, bword ptr [rbp-0x400]
       mov      rsi, qword ptr [rbp-0x3F8]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG99
       mov      rdi, bword ptr [rbp-0x58]
       mov      rsi, qword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG98
       mov      rdi, bword ptr [rbp+0x10]
       mov      rsi, qword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG100
 
G_M000_IG94:                ;; offset=0x1090
       mov      rdi, 0x7A8492761838
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG95:                ;; offset=0x109F
       mov      rdi, 0x7A849276183C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG96:                ;; offset=0x10AE
       xor      eax, eax
 
G_M000_IG97:                ;; offset=0x10B0
       add      rsp, 0x480
       pop      rbp
       ret      
 
G_M000_IG98:                ;; offset=0x10B9
       mov      rdi, 0x7A8492761840
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG95
 
G_M000_IG99:                ;; offset=0x10CA
       mov      rdi, 0x7A8492761844
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG95
 
G_M000_IG100:                ;; offset=0x10DB
       mov      rdi, 0x7A8492761848
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x410], rax
       mov      qword ptr [rbp-0x408], rdx
       lea      rdi, [rsp]
       lea      rsi, [rbp+0x10]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       mov      eax, dword ptr [rbp+0x68]
       mov      dword ptr [rsp+0x10], eax
       mov      eax, dword ptr [rbp+0x70]
       imul     eax, dword ptr [rbp+0x78]
       mov      dword ptr [rsp+0x18], eax
       mov      eax, dword ptr [rbp+0x80]
       mov      dword ptr [rsp+0x20], eax
       movzx    rax, byte  ptr [rbp+0x88]
       mov      dword ptr [rsp+0x28], eax
       mov      rdi, bword ptr [rbp-0x410]
       mov      rsi, qword ptr [rbp-0x408]
       mov      rdx, bword ptr [rbp+0x20]
       mov      rcx, qword ptr [rbp+0x28]
       mov      r8, bword ptr [rbp-0x58]
       mov      r9, qword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG101:                ;; offset=0x116F
       add      rsp, 0x480
       pop      rbp
       ret      
 
G_M000_IG102:                ;; offset=0x1178
       call     CORINFO_HELP_OVERFLOW
       int3     
 
; Total bytes of code 4478

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int) (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier-0 switched to FullOpts code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100
; No PGO data
; 0 inlinees with PGO data; 0 single block inlinees; 2 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 104
       lea      rbp, [rsp+0x90]
       xor      eax, eax
       mov      qword ptr [rbp-0x48], rax
       mov      qword ptr [rbp-0x50], rax
       mov      qword ptr [rbp-0x70], 0x1EAA0F60
       mov      r14, rdx
       mov      r11d, dword ptr [rbp+0x10]
       mov      ebx, dword ptr [rbp+0x18]
       mov      r15d, dword ptr [rbp+0x20]
       mov      r10d, dword ptr [rbp+0x28]
 
G_M000_IG02:                ;; offset=0x003A
       mov      eax, r15d
       cdq      
       idiv     edx:eax, ebx
       mov      r13d, eax
       lea      r12d, [2*r13-0x01]
       mov      eax, r15d
       cdq      
       idiv     edx:eax, ebx
       lea      edx, [2*rdx-0x01]
       mov      dword ptr [rbp-0x30], edx
       cmp      r10d, 8
       jne      SHORT G_M000_IG08
 
G_M000_IG03:                ;; offset=0x0061
       lea      eax, [r15+0x07]
       cdq      
       idiv     edx:eax, ebx
       cmp      eax, r13d
       jne      SHORT G_M000_IG08
 
G_M000_IG04:                ;; offset=0x006D
       test     r12d, r12d
       jl       SHORT G_M000_IG08
 
G_M000_IG05:                ;; offset=0x0072
       lea      edx, [r12+0x03]
       cmp      edx, r9d
       jge      SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x007C
       mov      r13d, dword ptr [rbp-0x30]
       test     r13d, r13d
       jl       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x0085
       lea      edx, [r13+0x11]
       cmp      edx, r11d
       jl       G_M000_IG35
 
G_M000_IG08:                ;; offset=0x0092
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rax, [rsp+0x20]
       vxorps   ymm0, ymm0, ymm0
       vmovdqu32 zmmword ptr [rax], zmm0
       vmovdqu32 zmmword ptr [rax+0x40], zmm0
       vmovdqu32 zmmword ptr [rax+0x80], zmm0
       vmovdqu32 zmmword ptr [rax+0xC0], zmm0
       vmovdqu32 zmmword ptr [rax+0x100], zmm0
       vmovdqu32 zmmword ptr [rax+0x140], zmm0
       vmovdqu32 zmmword ptr [rax+0x180], zmm0
       vmovdqu32 zmmword ptr [rax+0x1C0], zmm0
       mov      r13, rax
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rax, [rsp+0x20]
       vxorps   ymm0, ymm0, ymm0
       vmovdqu32 zmmword ptr [rax], zmm0
       vmovdqu32 zmmword ptr [rax+0x40], zmm0
       vmovdqu32 zmmword ptr [rax+0x80], zmm0
       vmovdqu32 zmmword ptr [rax+0xC0], zmm0
       vmovdqu32 zmmword ptr [rax+0x100], zmm0
       vmovdqu32 zmmword ptr [rax+0x140], zmm0
       vmovdqu32 zmmword ptr [rax+0x180], zmm0
       vmovdqu32 zmmword ptr [rax+0x1C0], zmm0
       mov      r12, rax
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rax, [rsp+0x20]
       vxorps   ymm0, ymm0, ymm0
       vmovdqu32 zmmword ptr [rax], zmm0
       vmovdqu32 zmmword ptr [rax+0x40], zmm0
       vmovdqu32 zmmword ptr [rax+0x80], zmm0
       vmovdqu32 zmmword ptr [rax+0xC0], zmm0
       vmovdqu32 zmmword ptr [rax+0x100], zmm0
       vmovdqu32 zmmword ptr [rax+0x140], zmm0
       vmovdqu32 zmmword ptr [rax+0x180], zmm0
       vmovdqu32 zmmword ptr [rax+0x1C0], zmm0
       mov      qword ptr [rbp-0x38], rax
       xor      eax, eax
       mov      bword ptr [rbp-0x68], r14
       jmp      SHORT G_M000_IG10
       align    [0 bytes for IG13]
 
G_M000_IG09:                ;; offset=0x0182
       mov      eax, dword ptr [rbp-0x3C]
       inc      eax
       cmp      eax, 8
       mov      ebx, dword ptr [rbp+0x18]
       mov      r15d, dword ptr [rbp+0x20]
       jge      G_M000_IG22
 
G_M000_IG10:                ;; offset=0x0197
       mov      dword ptr [rbp+0x20], r15d
       mov      dword ptr [rbp-0x3C], eax
       lea      r14d, [r15+rax]
       mov      eax, r14d
       cdq      
       idiv     edx:eax, ebx
       lea      edx, [2*rax-0x01]
       mov      dword ptr [rbp-0x40], edx
       mov      dword ptr [rbp+0x18], ebx
       mov      eax, r14d
       cdq      
       idiv     edx:eax, ebx
       lea      eax, [2*rdx-0x01]
       xor      edx, edx
       jmp      SHORT G_M000_IG12
 
G_M000_IG11:                ;; offset=0x01C6
       mov      ecx, dword ptr [rbp-0x58]
       mov      rdi, bword ptr [rbp-0x60]
       inc      edx
       cmp      edx, 4
       jge      SHORT G_M000_IG09
 
G_M000_IG12:                ;; offset=0x01D4
       xor      r14d, r14d
       mov      bword ptr [rbp-0x60], rdi
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG19
 
G_M000_IG13:                ;; offset=0x01E0
       xor      esi, esi
 
G_M000_IG14:                ;; offset=0x01E2
       movsxd   rcx, ecx
       shl      rcx, 2
       lea      r15, [rcx+r13]
       test     esi, esi
       jne      G_M000_IG33
 
G_M000_IG15:                ;; offset=0x01F5
       xor      ebx, ebx
 
G_M000_IG16:                ;; offset=0x01F7
       mov      dword ptr [r15], ebx
       add      rcx, r12
       test     esi, esi
       jne      G_M000_IG34
 
G_M000_IG17:                ;; offset=0x0205
       xor      esi, esi
 
G_M000_IG18:                ;; offset=0x0207
       mov      dword ptr [rcx], esi
       inc      r14d
       cmp      r14d, 4
       mov      dword ptr [rbp-0x3C], edi
       mov      esi, dword ptr [rbp-0x54]
       jge      SHORT G_M000_IG11
 
G_M000_IG19:                ;; offset=0x0218
       lea      ecx, [r14+4*rdx]
       mov      edi, dword ptr [rbp-0x3C]
       lea      ecx, [rdi+8*rcx]
       cmp      edi, r10d
       mov      dword ptr [rbp-0x54], esi
       jge      SHORT G_M000_IG13
 
G_M000_IG20:                ;; offset=0x022A
       mov      esi, dword ptr [rbp-0x40]
       lea      r15d, [rsi+rdx]
       cmp      r15d, r9d
       jae      SHORT G_M000_IG13
 
G_M000_IG21:                ;; offset=0x0236
       lea      r15d, [rax+r14]
       cmp      r15d, r11d
       setb     r15b
       movzx    r15, r15b
       mov      esi, r15d
       jmp      SHORT G_M000_IG14
 
G_M000_IG22:                ;; offset=0x024A
       mov      r14, bword ptr [rbp-0x68]
       xor      rax, rax
       test     esi, esi
       cmovne   rax, rdi
       mov      bword ptr [rbp-0x48], rax
       xor      rdi, rdi
       test     ecx, ecx
       cmovne   rdi, r14
       mov      bword ptr [rbp-0x50], rdi
       mov      rcx, rdi
       xor      edi, edi
       cmp      edi, r8d
       jge      G_M000_IG28
 
G_M000_IG23:                ;; offset=0x0274
       mov      esi, edi
       imul     esi, r9d
       imul     esi, r11d
       movsxd   rsi, esi
       lea      rsi, [rax+4*rsi]
       xor      r10d, r10d
       mov      ebx, 4
       align    [0 bytes for IG24]
 
G_M000_IG24:                ;; offset=0x028D
       lea      r15d, [8*r10]
       vxorps   ymm0, ymm0, ymm0
       movsxd   r15, r15d
       vmovups  ymm1, ymmword ptr [r13+4*r15]
       vmovups  ymm2, ymmword ptr [r12+4*r15]
       vmovaps  ymm3, ymm2
       vgatherdps ymm0, dword ptr [rsi+4*xmm1], ymm3
       vxorps   ymm1, ymm1, ymm1
       vmovups  ymm2, ymmword ptr [r13+4*r15+0x20]
       vmovups  ymm3, ymmword ptr [r12+4*r15+0x20]
       vmovaps  ymm4, ymm3
       vgatherdps ymm1, dword ptr [rsi+4*xmm2], ymm4
       vxorps   ymm2, ymm2, ymm2
       vmovups  ymm3, ymmword ptr [r13+4*r15+0x40]
       vmovups  ymm4, ymmword ptr [r12+4*r15+0x40]
       vmovaps  ymm5, ymm4
       vgatherdps ymm2, dword ptr [rsi+4*xmm3], ymm5
       vxorps   ymm3, ymm3, ymm3
       vmovups  ymm4, ymmword ptr [r13+4*r15+0x60]
       vmovups  ymm5, ymmword ptr [r12+4*r15+0x60]
       vmovaps  ymm6, ymm5
       vgatherdps ymm3, dword ptr [rsi+4*xmm4], ymm6
       vsubps   ymm0, ymm0, ymm2
       movsxd   r15, r10d
       shl      r15, 5
       mov      rdx, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [rdx+r15], ymm0
       vaddps   ymm0, ymm1, ymm2
       lea      r15d, [r10+0x01]
       movsxd   r15, r15d
       shl      r15, 5
       vmovups  ymmword ptr [rdx+r15], ymm0
       vsubps   ymm0, ymm2, ymm1
       lea      r15d, [r10+0x02]
       movsxd   r15, r15d
       shl      r15, 5
       vmovups  ymmword ptr [rdx+r15], ymm0
       vsubps   ymm0, ymm1, ymm3
       lea      r15d, [r10+0x03]
       movsxd   r15, r15d
       shl      r15, 5
       vmovups  ymmword ptr [rdx+r15], ymm0
       add      r10d, 4
       dec      ebx
       jne      G_M000_IG24
 
G_M000_IG25:                ;; offset=0x0367
       xor      esi, esi
       align    [0 bytes for IG26]
 
G_M000_IG26:                ;; offset=0x0369
       movsxd   r10, esi
       shl      r10, 5
       mov      rdx, qword ptr [rbp-0x38]
       vmovups  ymm0, ymmword ptr [rdx+r10]
       lea      r10d, [rsi+0x04]
       movsxd   rbx, r10d
       shl      rbx, 5
       vmovups  ymm1, ymmword ptr [rdx+rbx]
       lea      ebx, [rsi+0x08]
       movsxd   r15, ebx
       shl      r15, 5
       vmovups  ymm2, ymmword ptr [rdx+r15]
       lea      r15d, [rsi+0x0C]
       movsxd   r14, r15d
       shl      r14, 5
       vmovups  ymm3, ymmword ptr [rdx+r14]
       vsubps   ymm0, ymm0, ymm2
       mov      r14d, esi
       imul     r14d, r8d
       add      r14d, edi
       shl      r14d, 3
       movsxd   r14, r14d
       vmovups  ymmword ptr [rcx+4*r14], ymm0
       vaddps   ymm0, ymm1, ymm2
       imul     r10d, r8d
       add      r10d, edi
       shl      r10d, 3
       movsxd   r10, r10d
       vmovups  ymmword ptr [rcx+4*r10], ymm0
       vsubps   ymm0, ymm2, ymm1
       imul     ebx, r8d
       add      ebx, edi
       shl      ebx, 3
       movsxd   r10, ebx
       vmovups  ymmword ptr [rcx+4*r10], ymm0
       vsubps   ymm0, ymm1, ymm3
       imul     r15d, r8d
       add      r15d, edi
       shl      r15d, 3
       movsxd   r10, r15d
       vmovups  ymmword ptr [rcx+4*r10], ymm0
       inc      esi
       cmp      esi, 4
       jl       G_M000_IG26
 
G_M000_IG27:                ;; offset=0x0417
       inc      edi
       cmp      edi, r8d
       jl       G_M000_IG23
 
G_M000_IG28:                ;; offset=0x0422
       xor      eax, eax
       mov      bword ptr [rbp-0x48], rax
 
G_M000_IG29:                ;; offset=0x0428
       mov      bword ptr [rbp-0x50], rax
 
G_M000_IG30:                ;; offset=0x042C
       cmp      qword ptr [rbp-0x70], 0x1EAA0F60
       je       SHORT G_M000_IG31
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG31:                ;; offset=0x043B
       nop      
 
G_M000_IG32:                ;; offset=0x043C
       vzeroupper 
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG33:                ;; offset=0x044E
       mov      dword ptr [rbp-0x2C], r8d
       mov      ebx, dword ptr [rbp-0x40]
       lea      r8d, [rbx+rdx]
       imul     r8d, r11d
       add      r8d, eax
       add      r8d, r14d
       mov      ebx, r8d
       mov      r8d, dword ptr [rbp-0x2C]
       jmp      G_M000_IG16
 
G_M000_IG34:                ;; offset=0x046F
       mov      esi, -1
       jmp      G_M000_IG18
 
G_M000_IG35:                ;; offset=0x0479
       mov      dword ptr [rsp], r11d
       mov      dword ptr [rsp+0x08], r12d
       mov      dword ptr [rsp+0x10], r13d
       mov      rdx, r14
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int)]
       jmp      SHORT G_M000_IG30
 
; Total bytes of code 1170

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x9D0
       lea      rbp, [rsp+0x9D0]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa32 xmmword ptr [rbp-0x7B0], xmm8
       mov      rax, -0x750
       vmovdqa  xmmword ptr [rbp+rax-0x50], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x50], rax
       mov      qword ptr [rbp-0x30], rdi
       mov      qword ptr [rbp-0x38], rsi
       mov      qword ptr [rbp-0x40], rdx
       mov      dword ptr [rbp-0x44], ecx
       mov      dword ptr [rbp-0x48], r8d
 
G_M000_IG02:                ;; offset=0x0055
       mov      dword ptr [rbp-0x9C8], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x4C], eax
       jmp      G_M000_IG22
 
G_M000_IG03:                ;; offset=0x0069
       xor      eax, eax
       mov      dword ptr [rbp-0x50], eax
       jmp      G_M000_IG11
 
G_M000_IG04:                ;; offset=0x0073
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
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x3B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x3F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x430], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x470], zmm0
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       imul     eax, dword ptr [rbp-0x48]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       movsxd   rcx, dword ptr [rbp-0x50]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x478], rax
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x480], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x48C], eax
       jmp      G_M000_IG07
 
G_M000_IG05:                ;; offset=0x019F
       mov      rdi, 0x7A8492761908
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x478]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x4F0], zmm0
       mov      rax, qword ptr [rbp-0x478]
       vmovups  zmm0, zmmword ptr [rax+0x40]
       vmovups  zmmword ptr [rbp-0x530], zmm0
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x7F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x7F0]
       vmovups  zmm1, zmmword ptr [rbp-0xB0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0xB0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x7F0]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x04]
       vmovups  zmmword ptr [rbp-0x830], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x830]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x830]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x08]
       vmovups  zmmword ptr [rbp-0x870], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x870]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x870]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x0C]
       vmovups  zmmword ptr [rbp-0x8B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x8B0]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x8B0]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x10]
       vmovups  zmmword ptr [rbp-0x8F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x8F0]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
 
G_M000_IG06:                ;; offset=0x03B2
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x8F0]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x14]
       vmovups  zmmword ptr [rbp-0x930], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x930]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x930]
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x18]
       vmovups  zmmword ptr [rbp-0x970], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x970]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x970]
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x1C]
       vmovups  zmmword ptr [rbp-0x9B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x9B0]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x9B0]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       movsxd   rax, dword ptr [rbp-0x48]
       mov      rcx, qword ptr [rbp-0x478]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x478], rax
       mov      rax, qword ptr [rbp-0x480]
       add      rax, 32
       mov      qword ptr [rbp-0x480], rax
       mov      eax, dword ptr [rbp-0x48C]
       inc      eax
       mov      dword ptr [rbp-0x48C], eax
 
G_M000_IG07:                ;; offset=0x0552
       mov      eax, dword ptr [rbp-0x9C8]
       dec      eax
       mov      dword ptr [rbp-0x9C8], eax
       cmp      dword ptr [rbp-0x9C8], 0
       jg       SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x0569
       lea      rdi, [rbp-0x9C8]
       mov      esi, 487
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG09:                ;; offset=0x057A
       mov      eax, dword ptr [rbp-0x48C]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG05
       mov      rdi, 0x7A849276190C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x48]
       add      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x9B8], rax
       mov      rax, qword ptr [rbp-0x9B8]
       add      rax, 512
       mov      qword ptr [rbp-0x488], rax
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0xB0]
       vmovups  zmmword ptr [rax], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0xF0]
       vmovups  zmmword ptr [rax+0x40], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x130]
       vmovups  zmmword ptr [rax+0x80], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x170]
       vmovups  zmmword ptr [rax+0xC0], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x1B0]
       vmovups  zmmword ptr [rax+0x100], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x1F0]
       vmovups  zmmword ptr [rax+0x140], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x230]
       vmovups  zmmword ptr [rax+0x180], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x270]
       vmovups  zmmword ptr [rax+0x1C0], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x2B0]
       vmovups  zmmword ptr [rax], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rax+0x40], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x330]
       vmovups  zmmword ptr [rax+0x80], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x370]
       vmovups  zmmword ptr [rax+0xC0], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x3B0]
       vmovups  zmmword ptr [rax+0x100], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x3F0]
 
G_M000_IG10:                ;; offset=0x0711
       vmovups  zmmword ptr [rax+0x140], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x430]
       vmovups  zmmword ptr [rax+0x180], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x470]
       vmovups  zmmword ptr [rax+0x1C0], zmm0
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 32
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG11:                ;; offset=0x0751
       mov      eax, dword ptr [rbp-0x9C8]
       dec      eax
       mov      dword ptr [rbp-0x9C8], eax
       cmp      dword ptr [rbp-0x9C8], 0
       jg       SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x0768
       lea      rdi, [rbp-0x9C8]
       mov      esi, 742
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG13:                ;; offset=0x0779
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 32
       cmp      eax, dword ptr [rbp-0x48]
       jle      G_M000_IG04
       jmp      G_M000_IG19
 
G_M000_IG14:                ;; offset=0x078D
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x570], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x5B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x5F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x630], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x670], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x6B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x6F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x730], zmm0
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       imul     eax, dword ptr [rbp-0x48]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       movsxd   rcx, dword ptr [rbp-0x50]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x738], rax
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x740], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x744], eax
       jmp      G_M000_IG16
 
G_M000_IG15:                ;; offset=0x0849
       mov      rdi, 0x7A8492761910
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x738]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x7B0], zmm0
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x04]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x08]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x0C]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x630], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x10]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x14]
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x18]
       vmovups  zmm1, zmmword ptr [rbp-0x6F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x6F0], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x1C]
       vmovups  zmm1, zmmword ptr [rbp-0x730]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x730], zmm1
       movsxd   rax, dword ptr [rbp-0x48]
       mov      rcx, qword ptr [rbp-0x738]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x738], rax
       mov      rax, qword ptr [rbp-0x740]
       add      rax, 32
       mov      qword ptr [rbp-0x740], rax
       mov      eax, dword ptr [rbp-0x744]
       inc      eax
       mov      dword ptr [rbp-0x744], eax
 
G_M000_IG16:                ;; offset=0x0A04
       mov      eax, dword ptr [rbp-0x9C8]
       dec      eax
       mov      dword ptr [rbp-0x9C8], eax
       cmp      dword ptr [rbp-0x9C8], 0
       jg       SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x0A1B
       lea      rdi, [rbp-0x9C8]
       mov      esi, 0x42E
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG18:                ;; offset=0x0A2C
       mov      eax, dword ptr [rbp-0x744]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG15
       mov      rdi, 0x7A8492761914
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x48]
       add      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x9C0], rax
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x570]
       vmovups  zmmword ptr [rax], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x5B0]
       vmovups  zmmword ptr [rax+0x40], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x5F0]
       vmovups  zmmword ptr [rax+0x80], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x630]
       vmovups  zmmword ptr [rax+0xC0], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x670]
       vmovups  zmmword ptr [rax+0x100], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x6B0]
       vmovups  zmmword ptr [rax+0x140], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x6F0]
       vmovups  zmmword ptr [rax+0x180], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x730]
       vmovups  zmmword ptr [rax+0x1C0], zmm0
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 16
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG19:                ;; offset=0x0B30
       mov      eax, dword ptr [rbp-0x9C8]
       dec      eax
       mov      dword ptr [rbp-0x9C8], eax
       cmp      dword ptr [rbp-0x9C8], 0
       jg       SHORT G_M000_IG21
 
G_M000_IG20:                ;; offset=0x0B47
       lea      rdi, [rbp-0x9C8]
       mov      esi, 0x4B1
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG21:                ;; offset=0x0B58
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp-0x48]
       jl       G_M000_IG14
       mov      rdi, 0x7A8492761918
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       inc      eax
       mov      dword ptr [rbp-0x4C], eax
 
G_M000_IG22:                ;; offset=0x0B7B
       mov      eax, dword ptr [rbp-0x9C8]
       dec      eax
       mov      dword ptr [rbp-0x9C8], eax
       cmp      dword ptr [rbp-0x9C8], 0
       jg       SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x0B92
       lea      rdi, [rbp-0x9C8]
       mov      esi, 0x4BD
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG24:                ;; offset=0x0BA3
       cmp      dword ptr [rbp-0x4C], 16
       jl       G_M000_IG03
       mov      rdi, 0x7A849276191C
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG25:                ;; offset=0x0BBD
       vzeroupper 
       add      rsp, 0x9D0
       pop      rbp
       ret      
 
; Total bytes of code 3017

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
       mov      rdi, 0x7A8492761960
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
       mov      rdi, 0x7A8492761964
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
       inc      eax
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x338]
       vmovups  zmm0, zmmword ptr [rbp-0x330]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG09:                ;; offset=0x0475
       mov      rdi, 0x7A8492761968
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
       mov      rdi, 0x7A849276196C
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
       mov      rdi, 0x7A8492761970
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG16:                ;; offset=0x051C
       vzeroupper 
       add      rsp, 0x400
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x0528
       mov      rdi, 0x7A8492761974
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG09
 
; Total bytes of code 1340

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int) (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier-0 switched to FullOpts code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 180
; No PGO data
; 0 inlinees with PGO data; 0 single block inlinees; 2 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 24
       lea      rbp, [rsp+0x40]
       xor      eax, eax
       mov      qword ptr [rbp-0x30], rax
       mov      qword ptr [rbp-0x38], rax
       mov      qword ptr [rbp-0x40], 0x1EAA0F60
       mov      eax, dword ptr [rbp+0x10]
       mov      r10d, dword ptr [rbp+0x18]
       mov      r11d, dword ptr [rbp+0x20]
 
G_M000_IG02:                ;; offset=0x0030
       vmovups  ymm0, ymmword ptr [reloc @RWD00]
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rbx, [rsp]
       vxorps   ymm1, ymm1, ymm1
       vmovdqu32 zmmword ptr [rbx], zmm1
       vmovdqu32 zmmword ptr [rbx+0x40], zmm1
       vmovdqu32 zmmword ptr [rbx+0x80], zmm1
       vmovdqu32 zmmword ptr [rbx+0xC0], zmm1
       vmovdqu32 zmmword ptr [rbx+0x100], zmm1
       vmovdqu32 zmmword ptr [rbx+0x140], zmm1
       vmovdqu32 zmmword ptr [rbx+0x180], zmm1
       vmovdqu32 zmmword ptr [rbx+0x1C0], zmm1
       xor      r15, r15
       test     esi, esi
       cmovne   r15, rdi
       mov      bword ptr [rbp-0x30], r15
       mov      rdi, r15
       xor      rsi, rsi
       test     ecx, ecx
       cmovne   rsi, rdx
       mov      bword ptr [rbp-0x38], rsi
       mov      rcx, rsi
       xor      edx, edx
       cmp      edx, r8d
       jge      G_M000_IG08
 
G_M000_IG03:                ;; offset=0x00AB
       mov      esi, edx
       imul     esi, r9d
       imul     esi, eax
       movsxd   rsi, esi
       lea      rsi, [rdi+4*rsi]
       xor      r15d, r15d
       movsxd   r14, r11d
       shl      r14, 2
       align    [0 bytes for IG04]
 
G_M000_IG04:                ;; offset=0x00C5
       lea      r13d, [r10+r15]
       imul     r13d, eax
       movsxd   r13, r13d
       shl      r13, 2
       add      r13, rsi
       add      r13, r14
       vpermps  ymm1, ymm0, ymmword ptr [r13]
       vpermps  ymm2, ymm0, ymmword ptr [r13+0x20]
       vpermps  ymm3, ymm0, ymmword ptr [r13+0x08]
       vpermps  ymm4, ymm0, ymmword ptr [r13+0x28]
       vperm2f128 ymm5, ymm1, ymm2, 32
       vperm2f128 ymm1, ymm1, ymm2, 49
       vperm2f128 ymm2, ymm3, ymm4, 32
       vperm2f128 ymm3, ymm3, ymm4, 49
       vsubps   ymm4, ymm5, ymm2
       lea      r13d, [4*r15]
       movsxd   r12, r13d
       shl      r12, 5
       vmovups  ymmword ptr [rbx+r12], ymm4
       vaddps   ymm4, ymm1, ymm2
       lea      r12d, [r13+0x01]
       movsxd   r12, r12d
       shl      r12, 5
       vmovups  ymmword ptr [rbx+r12], ymm4
       vsubps   ymm2, ymm2, ymm1
       lea      r12d, [r13+0x02]
       movsxd   r12, r12d
       shl      r12, 5
       vmovups  ymmword ptr [rbx+r12], ymm2
       vsubps   ymm1, ymm1, ymm3
       add      r13d, 3
       movsxd   r13, r13d
       shl      r13, 5
       vmovups  ymmword ptr [rbx+r13], ymm1
       inc      r15d
       cmp      r15d, 4
       jl       G_M000_IG04
 
G_M000_IG05:                ;; offset=0x016F
       xor      esi, esi
       align    [0 bytes for IG06]
 
G_M000_IG06:                ;; offset=0x0171
       movsxd   r15, esi
       shl      r15, 5
       vmovups  ymm1, ymmword ptr [rbx+r15]
       lea      r15d, [rsi+0x04]
       movsxd   r14, r15d
       shl      r14, 5
       vmovups  ymm2, ymmword ptr [rbx+r14]
       lea      r14d, [rsi+0x08]
       movsxd   r13, r14d
       shl      r13, 5
       vmovups  ymm3, ymmword ptr [rbx+r13]
       lea      r13d, [rsi+0x0C]
       movsxd   r12, r13d
       shl      r12, 5
       vmovups  ymm4, ymmword ptr [rbx+r12]
       vsubps   ymm1, ymm1, ymm3
       mov      r12d, esi
       imul     r12d, r8d
       add      r12d, edx
       shl      r12d, 3
       movsxd   r12, r12d
       vmovups  ymmword ptr [rcx+4*r12], ymm1
       vaddps   ymm1, ymm2, ymm3
       imul     r15d, r8d
       add      r15d, edx
       shl      r15d, 3
       movsxd   r15, r15d
       vmovups  ymmword ptr [rcx+4*r15], ymm1
       vsubps   ymm1, ymm3, ymm2
       imul     r14d, r8d
       add      r14d, edx
       shl      r14d, 3
       movsxd   r15, r14d
       vmovups  ymmword ptr [rcx+4*r15], ymm1
       vsubps   ymm1, ymm2, ymm4
       imul     r13d, r8d
       add      r13d, edx
       shl      r13d, 3
       movsxd   r15, r13d
       vmovups  ymmword ptr [rcx+4*r15], ymm1
       inc      esi
       cmp      esi, 4
       jl       G_M000_IG06
 
G_M000_IG07:                ;; offset=0x021F
       inc      edx
       cmp      edx, r8d
       jl       G_M000_IG03
 
G_M000_IG08:                ;; offset=0x022A
       xor      eax, eax
       mov      bword ptr [rbp-0x30], rax
 
G_M000_IG09:                ;; offset=0x0230
       mov      bword ptr [rbp-0x38], rax
       cmp      qword ptr [rbp-0x40], 0x1EAA0F60
       je       SHORT G_M000_IG10
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG10:                ;; offset=0x0243
       nop      
 
G_M000_IG11:                ;; offset=0x0244
       vzeroupper 
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
RWD00  	dq	0000000200000000h, 0000000600000004h, 0000000300000001h, 0000000700000005h

; Total bytes of code 598

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 320
       lea      rbp, [rsp+0x140]
       vxorps   xmm8, xmm8, xmm8
       vmovdqu32 zmmword ptr [rbp-0x130], zmm8
       vmovdqu32 zmmword ptr [rbp-0xF0], zmm8
       vmovdqu32 zmmword ptr [rbp-0xB0], zmm8
       vmovdqu32 zmmword ptr [rbp-0x80], zmm8
       xor      eax, eax
       mov      qword ptr [rbp-0x40], rax
       mov      bword ptr [rbp-0x38], rdi
       mov      qword ptr [rbp-0x30], rsi
 
G_M000_IG02:                ;; offset=0x0048
       mov      dword ptr [rbp-0x138], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x3C], eax
       lea      rdi, [rbp-0x38]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0x50], rax
       mov      rax, bword ptr [rbp-0x50]
       mov      qword ptr [rbp-0x140], rax
       mov      rax, qword ptr [rbp-0x140]
       mov      qword ptr [rbp-0x48], rax
       vbroadcastss zmm0, dword ptr [reloc @RWD00]
       vmovups  zmmword ptr [rbp-0xB0], zmm0
       vbroadcastss zmm0, dword ptr [reloc @RWD04]
       vmovups  zmmword ptr [rbp-0xF0], zmm0
       jmp      SHORT G_M000_IG06
 
G_M000_IG03:                ;; offset=0x00A5
       movsxd   rax, dword ptr [rbp-0x3C]
       mov      rcx, qword ptr [rbp-0x48]
       vmovups  zmm0, zmmword ptr [rcx+4*rax]
       vpandd   zmm0, zmm0, zmmword ptr [rbp-0xB0]
       vcmpgtps k1, zmm0, zmmword ptr [rbp-0xF0]
       kmovw    eax, k1
       test     rax, rax
       je       SHORT G_M000_IG05
       mov      rdi, 0x7A8492764CF0
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG04:                ;; offset=0x00E3
       vzeroupper 
       add      rsp, 320
       pop      rbp
       ret      
 
G_M000_IG05:                ;; offset=0x00EF
       mov      rdi, 0x7A8492764CF4
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       add      eax, 16
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG06:                ;; offset=0x0107
       mov      eax, dword ptr [rbp-0x138]
       dec      eax
       mov      dword ptr [rbp-0x138], eax
       cmp      dword ptr [rbp-0x138], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x011E
       lea      rdi, [rbp-0x138]
       mov      esi, 93
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x012F
       mov      eax, dword ptr [rbp-0x30]
       add      eax, -16
       cmp      dword ptr [rbp-0x3C], eax
       jle      G_M000_IG03
       xor      eax, eax
       mov      bword ptr [rbp-0x50], rax
       jmp      SHORT G_M000_IG12
 
G_M000_IG09:                ;; offset=0x0146
       mov      eax, dword ptr [rbp-0x30]
       cmp      dword ptr [rbp-0x3C], eax
       jae      G_M000_IG16
       mov      eax, dword ptr [rbp-0x3C]
       mov      rcx, bword ptr [rbp-0x38]
       vmovss   xmm0, dword ptr [rcx+4*rax]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       jbe      SHORT G_M000_IG11
       mov      rdi, 0x7A8492764CF8
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG10:                ;; offset=0x0181
       vzeroupper 
       add      rsp, 320
       pop      rbp
       ret      
 
G_M000_IG11:                ;; offset=0x018D
       mov      rdi, 0x7A8492764CFC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       inc      eax
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG12:                ;; offset=0x01A4
       mov      eax, dword ptr [rbp-0x138]
       dec      eax
       mov      dword ptr [rbp-0x138], eax
       cmp      dword ptr [rbp-0x138], 0
       jg       SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x01BB
       lea      rdi, [rbp-0x138]
       mov      esi, 235
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG14:                ;; offset=0x01CC
       mov      eax, dword ptr [rbp-0x3C]
       cmp      eax, dword ptr [rbp-0x30]
       jl       G_M000_IG09
       mov      rdi, 0x7A8492764D00
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, 1
 
G_M000_IG15:                ;; offset=0x01EC
       vzeroupper 
       add      rsp, 320
       pop      rbp
       ret      
 
G_M000_IG16:                ;; offset=0x01F8
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	7E7FFFFFh		; 8.50706e+37
RWD08  	dd	00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 510

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x5d
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 1
; 0 inlinees with PGO data; 0 single block inlinees; 1 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       mov      rbp, rsp
       mov      eax, dword ptr [rbp+0x114]
       mov      rcx, qword ptr [rbp+0x108]
       vmovups  zmm0, zmmword ptr [rbp+0xA0]
       vmovups  zmm1, zmmword ptr [rbp+0x60]
 
G_M000_IG02:                ;; offset=0x0029
       mov      rdx, bword ptr [rbp+0x118]
       mov      edi, dword ptr [rbp+0x120]
       lea      esi, [rdi-0x10]
       cmp      eax, esi
       jg       SHORT G_M000_IG05
       align    [0 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x003D
       movsxd   r8, eax
       vpandd   zmm2, zmm0, zmmword ptr [rcx+4*r8]
       vcmpgtps k1, zmm2, zmm1
       kmovw    r8d, k1
       test     r8, r8
       jne      SHORT G_M000_IG12
 
G_M000_IG04:                ;; offset=0x0057
       add      eax, 16
       cmp      eax, esi
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x005E
       xor      ecx, ecx
       mov      bword ptr [rbp+0x100], rcx
       cmp      eax, edi
       jge      SHORT G_M000_IG10
 
G_M000_IG06:                ;; offset=0x006B
       test     eax, eax
       jl       SHORT G_M000_IG14
 
G_M000_IG07:                ;; offset=0x006F
       vmovss   xmm0, dword ptr [reloc @RWD00]
 
G_M000_IG08:                ;; offset=0x0077
       mov      ecx, eax
       vmovss   xmm1, dword ptr [rdx+4*rcx]
       vandps   xmm1, xmm1, xmmword ptr [reloc @RWD16]
       vucomiss xmm1, xmm0
       ja       SHORT G_M000_IG12
 
G_M000_IG09:                ;; offset=0x008C
       inc      eax
       cmp      eax, edi
       jl       SHORT G_M000_IG08
 
G_M000_IG10:                ;; offset=0x0092
       mov      eax, 1
 
G_M000_IG11:                ;; offset=0x0097
       vzeroupper 
       add      rsp, 336
       pop      rbp
       ret      
 
G_M000_IG12:                ;; offset=0x00A3
       xor      eax, eax
 
G_M000_IG13:                ;; offset=0x00A5
       vzeroupper 
       add      rsp, 336
       pop      rbp
       ret      
 
G_M000_IG14:                ;; offset=0x00B1
       cmp      eax, edi
       jae      SHORT G_M000_IG16
       mov      ecx, eax
       vmovss   xmm1, dword ptr [rdx+4*rcx]
       vandps   xmm1, xmm1, xmmword ptr [reloc @RWD16]
       vmovss   xmm0, dword ptr [reloc @RWD00]
       vucomiss xmm1, xmm0
       ja       SHORT G_M000_IG12
 
G_M000_IG15:                ;; offset=0x00D2
       inc      eax
       cmp      eax, edi
       jl       SHORT G_M000_IG14
       jmp      SHORT G_M000_IG10
 
G_M000_IG16:                ;; offset=0x00DA
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7E7FFFFFh		; 8.50706e+37
RWD04  	dd	00000000h, 00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 224

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x3af
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 5
; 8 inlinees with PGO data; 165 single block inlinees; 37 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 208
       mov      qword ptr [rsp+0x558], r15
       mov      qword ptr [rsp+0x550], r14
       mov      qword ptr [rsp+0x548], r13
       mov      qword ptr [rsp+0x540], r12
       mov      qword ptr [rsp+0x538], rbx
       lea      rbp, [rsp+0xD0]
       xor      eax, eax
       mov      qword ptr [rbp-0x30], rax
       mov      qword ptr [rbp-0x38], rax
       mov      r13d, dword ptr [rbp+0x4F0]
       mov      r15d, dword ptr [rbp+0x4F8]
       mov      r14d, dword ptr [rbp+0x500]
       mov      ebx, dword ptr [rbp+0x508]
       mov      r12d, dword ptr [rbp+0x510]
       mov      eax, dword ptr [rbp+0x41C]
       mov      r11d, dword ptr [rbp+0x418]
       mov      r10d, dword ptr [rbp+0x414]
 
G_M000_IG02:                ;; offset=0x007C
       mov      r9, bword ptr [rbp+0x448]
       mov      bword ptr [rbp-0x68], r9
       mov      r8d, dword ptr [rbp+0x450]
       mov      dword ptr [rbp-0x40], r8d
       mov      rcx, bword ptr [rbp+0x438]
       mov      bword ptr [rbp-0x70], rcx
       mov      edx, dword ptr [rbp+0x440]
       mov      dword ptr [rbp-0x44], edx
       mov      rsi, bword ptr [rbp+0x4C0]
       mov      bword ptr [rbp-0x78], rsi
       mov      edi, dword ptr [rbp+0x4C8]
       mov      dword ptr [rbp-0x48], edi
       mov      rcx, bword ptr [rbp+0x4D0]
       mov      bword ptr [rbp-0x80], rcx
       mov      edx, dword ptr [rbp+0x4D8]
       mov      dword ptr [rbp-0x4C], edx
       mov      r9, bword ptr [rbp+0x4E0]
       mov      bword ptr [rbp-0x88], r9
       mov      r9d, dword ptr [rbp+0x4E8]
       mov      dword ptr [rbp-0x50], r9d
       mov      r8, bword ptr [rbp+0x458]
       mov      bword ptr [rbp-0x90], r8
       mov      r8d, dword ptr [rbp+0x460]
       mov      dword ptr [rbp-0x54], r8d
       mov      r8, bword ptr [rbp+0x4A0]
       mov      bword ptr [rbp-0x98], r8
       mov      r8d, dword ptr [rbp+0x4A8]
       mov      dword ptr [rbp-0x58], r8d
       mov      r8, bword ptr [rbp+0x4B0]
       mov      bword ptr [rbp-0xA0], r8
       mov      r8d, dword ptr [rbp+0x4B8]
       mov      dword ptr [rbp-0x5C], r8d
       mov      r8d, r14d
       imul     r8d, ebx
       mov      dword ptr [rbp-0x60], r8d
       cmp      r10d, r11d
       jl       G_M000_IG32
       jmp      G_M000_IG69
 
G_M000_IG03:                ;; offset=0x014B
       jmp      G_M000_IG34
 
G_M000_IG04:                ;; offset=0x0150
       xor      edi, edi
       movsxd   rsi, r15d
       shl      rsi, 2
       jmp      G_M000_IG13
       align    [0 bytes for IG10]
 
G_M000_IG05:                ;; offset=0x015E
       mov      dword ptr [rbp+0x4F0], r13d
 
G_M000_IG06:                ;; offset=0x0165
       mov      ecx, edi
       imul     ecx, r15d
       add      ecx, r9d
       shl      ecx, 3
       movsxd   rcx, ecx
       lea      rcx, [rdx+4*rcx]
       vmovups  ymmword ptr [rcx], ymm0
       vmovups  ymmword ptr [rcx+0x20], ymm1
       vmovups  ymmword ptr [rcx+0x40], ymm2
       vmovups  ymmword ptr [rcx+0x60], ymm3
       vmovups  ymmword ptr [rcx+0x80], ymm4
       vmovups  ymmword ptr [rcx+0xA0], ymm5
       vmovups  ymmword ptr [rcx+0xC0], ymm6
       vmovups  ymmword ptr [rcx+0xE0], ymm7
       add      r9d, 8
       cmp      r9d, r15d
       jge      G_M000_IG12
 
G_M000_IG07:                ;; offset=0x01B4
       mov      rcx, qword ptr [rbp+0x400]
       mov      r13d, dword ptr [rbp+0x4F0]
 
G_M000_IG08:                ;; offset=0x01C2
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r12d, edi
       imul     r12d, r13d
       mov      r10d, r12d
       imul     r10d, r15d
       movsxd   r10, r10d
       shl      r10, 2
       add      r10, rcx
       movsxd   rcx, r9d
       lea      rcx, [r10+4*rcx]
       shl      r12d, 3
       movsxd   r10, r12d
       mov      r12, qword ptr [rbp+0x408]
       lea      r10, [r12+4*r10]
       test     r13d, r13d
       jle      G_M000_IG05
 
G_M000_IG09:                ;; offset=0x021C
       mov      dword ptr [rbp+0x4F0], r13d
       mov      r12d, r13d
 
G_M000_IG10:                ;; offset=0x0226
       vmovups  ymm8, ymmword ptr [rcx]
       vfmadd231ps ymm0, ymm8, dword ptr [r10] {1to8}
       vfmadd231ps ymm1, ymm8, dword ptr [r10+0x04] {1to8}
       vfmadd231ps ymm2, ymm8, dword ptr [r10+0x08] {1to8}
       vfmadd231ps ymm3, ymm8, dword ptr [r10+0x0C] {1to8}
       vfmadd231ps ymm4, ymm8, dword ptr [r10+0x10] {1to8}
       vfmadd231ps ymm5, ymm8, dword ptr [r10+0x14] {1to8}
       vfmadd231ps ymm6, ymm8, dword ptr [r10+0x18] {1to8}
       vfmadd231ps ymm7, ymm8, dword ptr [r10+0x1C] {1to8}
       add      rcx, rsi
       add      r10, 32
       dec      r12d
       jne      SHORT G_M000_IG10
       jmp      G_M000_IG06
 
G_M000_IG11:                ;; offset=0x0272
       mov      dword ptr [rbp+0x4F0], r13d
 
G_M000_IG12:                ;; offset=0x0279
       inc      edi
       cmp      edi, 16
       mov      rcx, qword ptr [rbp+0x400]
       mov      r13d, dword ptr [rbp+0x4F0]
       jge      SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x028E
       xor      r9d, r9d
       cmp      r9d, r15d
       jl       G_M000_IG08
       jmp      SHORT G_M000_IG11
 
G_M000_IG14:                ;; offset=0x029C
       jmp      G_M000_IG51
       align    [0 bytes for IG19]
 
G_M000_IG15:                ;; offset=0x02A1
       mov      edx, dword ptr [rbp+0x410]
       mov      r12, qword ptr [rbp+0x3F0]
       mov      r13d, dword ptr [rbp+0x414]
 
G_M000_IG16:                ;; offset=0x02B5
       add      esi, 16
       cmp      esi, r15d
       jge      G_M000_IG27
 
G_M000_IG17:                ;; offset=0x02C1
       mov      dword ptr [rbp+0x414], r13d
       mov      r13d, dword ptr [rbp+0x4F0]
 
G_M000_IG18:                ;; offset=0x02CF
       xor      ecx, ecx
       cmp      ecx, dword ptr [rbp+0x410]
       mov      dword ptr [rbp+0x4F0], r13d
       jge      SHORT G_M000_IG15
 
G_M000_IG19:                ;; offset=0x02E0
       lea      r10d, [8*rsi]
       movsxd   r10, r10d
       shl      r10, 2
       add      r10, qword ptr [rbp+0x3F8]
       mov      r12d, ecx
       shl      r12d, 4
       movsxd   r12, r12d
       lea      r10, [r10+4*r12]
       lea      r12d, [8*r15]
       mov      r13d, dword ptr [rbp+0x414]
       lea      r11d, [rcx+r13]
       mov      r9d, dword ptr [rbp+0x41C]
       mov      eax, r11d
       cdq      
       idiv     edx:eax, r9d
       lea      edx, [rax+rax]
       mov      dword ptr [rbp-0x3C], edx
       mov      dword ptr [rbp+0x41C], r9d
       mov      eax, r11d
       cdq      
       idiv     edx:eax, r9d
       add      edx, edx
       vmovups  zmm0, zmmword ptr [r10]
       lea      eax, [4*r12]
       cdqe     
       vmovups  zmm1, zmmword ptr [r10+4*rax]
       vaddps   zmm0, zmm0, zmm1
       lea      eax, [8*r12]
       cdqe     
       vmovups  zmm2, zmmword ptr [r10+4*rax]
       vaddps   zmm0, zmm0, zmm2
       vsubps   zmm1, zmm1, zmm2
       lea      eax, [r12+2*r12]
       lea      r11d, [4*rax]
       movsxd   r11, r11d
       vsubps   zmm1, zmm1, zmmword ptr [r10+4*r11]
       movsxd   r11, r12d
       vmovups  zmm2, zmmword ptr [r10+4*r11]
       lea      r11d, [r12+4*r12]
       movsxd   r9, r11d
       vmovups  zmm3, zmmword ptr [r10+4*r9]
       vaddps   zmm2, zmm2, zmm3
       lea      r9d, [r12+8*r12]
       movsxd   r9, r9d
       vmovups  zmm4, zmmword ptr [r10+4*r9]
       vaddps   zmm2, zmm2, zmm4
       vsubps   zmm3, zmm3, zmm4
       imul     r9d, r12d, 13
       movsxd   r9, r9d
       vsubps   zmm3, zmm3, zmmword ptr [r10+4*r9]
       lea      r9d, [r12+r12]
       movsxd   r9, r9d
       vmovups  zmm4, zmmword ptr [r10+4*r9]
       lea      r9d, [rax+rax]
       movsxd   r9, r9d
       vmovups  zmm5, zmmword ptr [r10+4*r9]
       vaddps   zmm4, zmm4, zmm5
       add      r11d, r11d
       movsxd   r9, r11d
       vmovups  zmm6, zmmword ptr [r10+4*r9]
       vaddps   zmm4, zmm6, zmm4
       vsubps   zmm5, zmm5, zmm6
       imul     r9d, r12d, 14
       movsxd   r9, r9d
       vsubps   zmm5, zmm5, zmmword ptr [r10+4*r9]
       movsxd   r9, eax
 
G_M000_IG20:                ;; offset=0x041D
       vmovups  zmm6, zmmword ptr [r10+4*r9]
       lea      r9d, [8*r12]
       sub      r9d, r12d
       movsxd   r9, r9d
       vmovups  zmm7, zmmword ptr [r10+4*r9]
       vaddps   zmm6, zmm6, zmm7
       imul     r9d, r12d, 11
       movsxd   r9, r9d
       vmovups  zmm8, zmmword ptr [r10+4*r9]
       vaddps   zmm6, zmm6, zmm8
       vsubps   zmm7, zmm7, zmm8
       mov      r9d, r12d
       shl      r9d, 4
       sub      r9d, r12d
       movsxd   r9, r9d
       vsubps   zmm7, zmm7, zmmword ptr [r10+4*r9]
       vaddps   zmm0, zmm2, zmm0
       vaddps   zmm0, zmm0, zmm4
       vsubps   zmm2, zmm2, zmm4
       vsubps   zmm2, zmm2, zmm6
       vaddps   zmm1, zmm1, zmm3
       vaddps   zmm1, zmm1, zmm5
       vsubps   zmm3, zmm3, zmm5
       vsubps   zmm3, zmm3, zmm7
       mov      r9d, dword ptr [rbp-0x3C]
       mov      eax, r9d
       imul     eax, ebx
       add      eax, edx
       shl      eax, 4
       cdqe     
       mov      r10d, esi
       imul     r10d, edi
       movsxd   r10, r10d
       shl      r10, 2
       mov      r12, qword ptr [rbp+0x3F0]
       add      r10, r12
       lea      rax, [r10+4*rax]
       vmovups  zmmword ptr [rax], zmm0
       lea      r10d, [rdx+0x01]
       cmp      r10d, ebx
       jge      SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x04D9
       vmovups  zmmword ptr [rax+0x40], zmm2
 
G_M000_IG22:                ;; offset=0x04E0
       inc      r9d
       cmp      r9d, r14d
       jge      SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x04E8
       mov      r9d, ebx
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rax+4*r9], zmm1
       inc      edx
       cmp      edx, ebx
       jge      SHORT G_M000_IG24
       lea      edx, [rbx+0x01]
       shl      edx, 4
       movsxd   rdx, edx
       vmovups  zmmword ptr [rax+4*rdx], zmm3
 
G_M000_IG24:                ;; offset=0x050F
       inc      ecx
       mov      edx, dword ptr [rbp+0x410]
       cmp      ecx, edx
       jge      G_M000_IG16
 
G_M000_IG25:                ;; offset=0x051F
       mov      dword ptr [rbp+0x414], r13d
       jmp      G_M000_IG19
 
G_M000_IG26:                ;; offset=0x052B
       mov      dword ptr [rbp+0x4F0], r13d
       mov      r13d, dword ptr [rbp+0x414]
 
G_M000_IG27:                ;; offset=0x0539
       xor      edi, edi
       mov      bword ptr [rbp+0x3E8], rdi
 
G_M000_IG28:                ;; offset=0x0542
       mov      bword ptr [rbp+0x3E0], rdi
 
G_M000_IG29:                ;; offset=0x0549
       mov      bword ptr [rbp+0x3D8], rdi
 
G_M000_IG30:                ;; offset=0x0550
       mov      bword ptr [rbp+0x3D0], rdi
       add      r13d, 8
       mov      edi, dword ptr [rbp+0x418]
       cmp      r13d, edi
       mov      r10d, r13d
       mov      r11d, edi
       mov      eax, dword ptr [rbp+0x41C]
       jge      G_M000_IG61
 
G_M000_IG31:                ;; offset=0x0576
       mov      r12d, dword ptr [rbp+0x510]
       mov      r13d, dword ptr [rbp+0x4F0]
 
G_M000_IG32:                ;; offset=0x0584
       mov      dword ptr [rbp+0x418], r11d
       mov      r8d, r11d
       sub      r8d, r10d
       cmp      r8d, 8
       jl       G_M000_IG03
 
G_M000_IG33:                ;; offset=0x059B
       mov      r8d, 8
 
G_M000_IG34:                ;; offset=0x05A1
       mov      dword ptr [rbp+0x410], r8d
       mov      dword ptr [rsp], ebx
       mov      dword ptr [rbp+0x41C], eax
       mov      dword ptr [rsp+0x08], eax
       mov      dword ptr [rbp+0x414], r10d
       mov      dword ptr [rsp+0x10], r10d
       mov      dword ptr [rsp+0x18], r8d
       mov      rdi, bword ptr [rbp-0x90]
       mov      esi, dword ptr [rbp-0x54]
       mov      rdx, bword ptr [rbp-0x78]
       mov      ecx, dword ptr [rbp-0x48]
       mov      r8d, r13d
       mov      r9d, r14d
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      eax, dword ptr [rbp-0x48]
       mov      edi, eax
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG35:                ;; offset=0x05F7
       test     edi, edi
       je       SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x05FB
       mov      r9, bword ptr [rbp-0x78]
       mov      rsi, r9
 
G_M000_IG37:                ;; offset=0x0602
       mov      bword ptr [rbp-0x30], rsi
       xor      edi, edi
       cmp      eax, 8
       jl       SHORT G_M000_IG39
       align    [0 bytes for IG38]
 
G_M000_IG38:                ;; offset=0x060D
       mov      edx, edi
       sar      edx, 31
       and      edx, 7
       add      edx, edi
       sar      edx, 3
       movsxd   rdx, edx
       shl      rdx, 5
       vpand    ymm1, ymm0, ymmword ptr [rdx+rsi]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG70
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, eax
       jle      SHORT G_M000_IG38
       align    [0 bytes for IG39]
 
G_M000_IG39:                ;; offset=0x063F
       cmp      edi, eax
       jl       G_M000_IG71
       xor      edi, edi
       mov      bword ptr [rbp-0x30], rdi
       mov      edi, 1
 
G_M000_IG40:                ;; offset=0x0652
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       test     edi, edi
       je       G_M000_IG67
 
G_M000_IG41:                ;; offset=0x0660
       xor      rdi, rdi
       test     eax, eax
       je       SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x0666
       mov      r9, bword ptr [rbp-0x78]
       mov      rdi, r9
 
G_M000_IG43:                ;; offset=0x066D
       mov      bword ptr [rbp+0x3E8], rdi
       mov      qword ptr [rbp+0x408], rdi
       xor      rsi, rsi
       cmp      dword ptr [rbp-0x40], 0
       je       SHORT G_M000_IG45
 
G_M000_IG44:                ;; offset=0x0683
       mov      r8, bword ptr [rbp-0x68]
       mov      rsi, r8
 
G_M000_IG45:                ;; offset=0x068A
       mov      bword ptr [rbp+0x3E0], rsi
       mov      rcx, rsi
       mov      qword ptr [rbp+0x400], rcx
       xor      rdx, rdx
       cmp      dword ptr [rbp-0x4C], 0
       je       SHORT G_M000_IG47
 
G_M000_IG46:                ;; offset=0x06A3
       mov      rdx, bword ptr [rbp-0x80]
       mov      r11, rdx
       mov      rdx, r11
 
G_M000_IG47:                ;; offset=0x06AD
       mov      bword ptr [rbp+0x3D8], rdx
       mov      qword ptr [rbp+0x3F8], rdx
       xor      r11, r11
       cmp      dword ptr [rbp-0x50], 0
       je       SHORT G_M000_IG49
 
G_M000_IG48:                ;; offset=0x06C4
       mov      r11, bword ptr [rbp-0x88]
 
G_M000_IG49:                ;; offset=0x06CB
       mov      bword ptr [rbp+0x3D0], r11
       mov      qword ptr [rbp+0x3F0], r11
       mov      dword ptr [rbp+0x510], r12d
       cmp      r12d, 16
       jne      G_M000_IG04
 
G_M000_IG50:                ;; offset=0x06EA
       mov      ecx, r13d
       mov      r8d, r15d
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int)]
 
G_M000_IG51:                ;; offset=0x06F6
       mov      r8d, dword ptr [rbp-0x4C]
       mov      edi, r8d
       xor      rsi, rsi
       mov      bword ptr [rbp-0x38], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG52:                ;; offset=0x070C
       test     edi, edi
       je       SHORT G_M000_IG54
 
G_M000_IG53:                ;; offset=0x0710
       mov      r10, bword ptr [rbp-0x80]
       mov      rsi, r10
 
G_M000_IG54:                ;; offset=0x0717
       mov      bword ptr [rbp-0x38], rsi
       xor      edi, edi
       cmp      r8d, 8
       jl       SHORT G_M000_IG56
       align    [0 bytes for IG55]
 
G_M000_IG55:                ;; offset=0x0723
       mov      ecx, edi
       sar      ecx, 31
       and      ecx, 7
       add      ecx, edi
       sar      ecx, 3
       movsxd   rcx, ecx
       shl      rcx, 5
       vpand    ymm1, ymm0, ymmword ptr [rcx+rsi]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG72
       add      edi, 8
       lea      ecx, [rdi+0x08]
       cmp      ecx, r8d
       jle      SHORT G_M000_IG55
       align    [0 bytes for IG56]
 
G_M000_IG56:                ;; offset=0x0756
       cmp      edi, r8d
       jl       G_M000_IG73
       xor      edi, edi
       mov      bword ptr [rbp-0x38], rdi
       mov      edi, 1
 
G_M000_IG57:                ;; offset=0x076A
       xor      rsi, rsi
       mov      bword ptr [rbp-0x38], rsi
       test     edi, edi
       je       G_M000_IG67
 
G_M000_IG58:                ;; offset=0x0778
       cmp      dword ptr [rbp+0x510], 16
       jne      SHORT G_M000_IG60
 
G_M000_IG59:                ;; offset=0x0781
       mov      r9d, dword ptr [rbp-0x60]
       mov      edi, r9d
       xor      esi, esi
       cmp      esi, r15d
       jl       G_M000_IG18
       jmp      G_M000_IG26
 
G_M000_IG60:                ;; offset=0x0798
       mov      rdx, qword ptr [rbp+0x3F8]
       mov      r11, qword ptr [rbp+0x3F0]
       mov      r9d, dword ptr [rbp+0x414]
       mov      dword ptr [rsp], r9d
       mov      edi, dword ptr [rbp+0x410]
       mov      dword ptr [rsp+0x08], edi
       mov      rdi, rdx
       mov      rsi, r11
       mov      edx, r15d
       mov      ecx, r14d
       mov      r8d, ebx
       mov      r9d, dword ptr [rbp+0x41C]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
       mov      dword ptr [rbp+0x4F0], r13d
       mov      r13d, dword ptr [rbp+0x414]
       jmp      G_M000_IG27
 
G_M000_IG61:                ;; offset=0x07EA
       mov      rdi, bword ptr [rbp-0x88]
       mov      esi, dword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG67
 
G_M000_IG62:                ;; offset=0x0802
       mov      rdi, bword ptr [rbp-0x88]
       mov      esi, dword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG67
 
G_M000_IG63:                ;; offset=0x081A
       mov      rdi, bword ptr [rbp-0x70]
       mov      esi, dword ptr [rbp-0x44]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG67
 
G_M000_IG64:                ;; offset=0x082F
       mov      rdi, bword ptr [rbp-0x98]
       mov      esi, dword ptr [rbp-0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG67
 
G_M000_IG65:                ;; offset=0x0847
       lea      rdi, [rsp]
       lea      rsi, [rbp+0x4A0]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       mov      dword ptr [rsp+0x10], r15d
       mov      ebx, dword ptr [rbp-0x60]
       mov      dword ptr [rsp+0x18], ebx
       mov      r12d, dword ptr [rbp+0x510]
       mov      dword ptr [rsp+0x20], r12d
       movzx    r8, byte  ptr [rbp+0x518]
       mov      dword ptr [rsp+0x28], r8d
       mov      r8, bword ptr [rbp-0x70]
       mov      r9d, dword ptr [rbp-0x44]
       mov      rdx, bword ptr [rbp-0xA0]
       mov      ecx, dword ptr [rbp-0x5C]
       mov      rdi, bword ptr [rbp-0x88]
       mov      esi, dword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG66:                ;; offset=0x08AF
       vzeroupper 
       add      rsp, 0x538
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG67:                ;; offset=0x08C4
       xor      eax, eax
 
G_M000_IG68:                ;; offset=0x08C6
       vzeroupper 
       add      rsp, 0x538
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG69:                ;; offset=0x08DB
       mov      dword ptr [rbp+0x510], r12d
       jmp      G_M000_IG61
 
G_M000_IG70:                ;; offset=0x08E7
       xor      edi, edi
       jmp      G_M000_IG40
 
G_M000_IG71:                ;; offset=0x08EE
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      ecx, 0x7F800000
       andn     edx, edx, ecx
       je       SHORT G_M000_IG70
       inc      edi
       jmp      G_M000_IG39
 
G_M000_IG72:                ;; offset=0x0907
       xor      edi, edi
       jmp      G_M000_IG57
 
G_M000_IG73:                ;; offset=0x090E
       movsxd   rcx, edi
       mov      ecx, dword ptr [rsi+4*rcx]
       mov      r9d, 0x7F800000
       andn     ecx, ecx, r9d
       je       SHORT G_M000_IG74
       inc      edi
       mov      r8d, dword ptr [rbp-0x4C]
       jmp      G_M000_IG56
 
G_M000_IG74:                ;; offset=0x092C
       mov      r8d, dword ptr [rbp-0x4C]
       jmp      SHORT G_M000_IG72
 
RWD00  	dd	7F800000h		;       inf

; Total bytes of code 2354

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
       mov      rdi, 0x7A8492761960
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
       mov      rdi, 0x7A8492761964
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
       inc      eax
       shl      eax, 4
       cdqe     
       mov      rcx, qword ptr [rbp-0x338]
       vmovups  zmm0, zmmword ptr [rbp-0x330]
       vmovups  zmmword ptr [rcx+4*rax], zmm0
 
G_M000_IG09:                ;; offset=0x0475
       mov      rdi, 0x7A8492761968
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
       mov      rdi, 0x7A849276196C
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
       mov      rdi, 0x7A8492761970
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG16:                ;; offset=0x051C
       vzeroupper 
       add      rsp, 0x400
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x0528
       mov      rdi, 0x7A8492761974
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG09
 
; Total bytes of code 1340

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x9D0
       lea      rbp, [rsp+0x9D0]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa32 xmmword ptr [rbp-0x7B0], xmm8
       mov      rax, -0x750
       vmovdqa  xmmword ptr [rbp+rax-0x50], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x50], rax
       mov      qword ptr [rbp-0x30], rdi
       mov      qword ptr [rbp-0x38], rsi
       mov      qword ptr [rbp-0x40], rdx
       mov      dword ptr [rbp-0x44], ecx
       mov      dword ptr [rbp-0x48], r8d
 
G_M000_IG02:                ;; offset=0x0055
       mov      dword ptr [rbp-0x9C8], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x4C], eax
       jmp      G_M000_IG22
 
G_M000_IG03:                ;; offset=0x0069
       xor      eax, eax
       mov      dword ptr [rbp-0x50], eax
       jmp      G_M000_IG11
 
G_M000_IG04:                ;; offset=0x0073
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
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x3B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x3F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x430], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x470], zmm0
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       imul     eax, dword ptr [rbp-0x48]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       movsxd   rcx, dword ptr [rbp-0x50]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x478], rax
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x480], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x48C], eax
       jmp      G_M000_IG07
 
G_M000_IG05:                ;; offset=0x019F
       mov      rdi, 0x7A8492761908
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x478]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x4F0], zmm0
       mov      rax, qword ptr [rbp-0x478]
       vmovups  zmm0, zmmword ptr [rax+0x40]
       vmovups  zmmword ptr [rbp-0x530], zmm0
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmmword ptr [rbp-0x7F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x7F0]
       vmovups  zmm1, zmmword ptr [rbp-0xB0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0xB0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x7F0]
       vmovups  zmm1, zmmword ptr [rbp-0x2B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x2B0], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x04]
       vmovups  zmmword ptr [rbp-0x830], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x830]
       vmovups  zmm1, zmmword ptr [rbp-0xF0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0xF0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x830]
       vmovups  zmm1, zmmword ptr [rbp-0x2F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x2F0], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x08]
       vmovups  zmmword ptr [rbp-0x870], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x870]
       vmovups  zmm1, zmmword ptr [rbp-0x130]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0x130], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x870]
       vmovups  zmm1, zmmword ptr [rbp-0x330]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x330], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x0C]
       vmovups  zmmword ptr [rbp-0x8B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x8B0]
       vmovups  zmm1, zmmword ptr [rbp-0x170]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0x170], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x8B0]
       vmovups  zmm1, zmmword ptr [rbp-0x370]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x370], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x10]
       vmovups  zmmword ptr [rbp-0x8F0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x8F0]
       vmovups  zmm1, zmmword ptr [rbp-0x1B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
 
G_M000_IG06:                ;; offset=0x03B2
       vmovups  zmmword ptr [rbp-0x1B0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x8F0]
       vmovups  zmm1, zmmword ptr [rbp-0x3B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x3B0], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x14]
       vmovups  zmmword ptr [rbp-0x930], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x930]
       vmovups  zmm1, zmmword ptr [rbp-0x1F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0x1F0], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x930]
       vmovups  zmm1, zmmword ptr [rbp-0x3F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x3F0], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x18]
       vmovups  zmmword ptr [rbp-0x970], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x970]
       vmovups  zmm1, zmmword ptr [rbp-0x230]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0x230], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x970]
       vmovups  zmm1, zmmword ptr [rbp-0x430]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x430], zmm1
       mov      rax, qword ptr [rbp-0x480]
       vbroadcastss zmm0, dword ptr [rax+0x1C]
       vmovups  zmmword ptr [rbp-0x9B0], zmm0
       vmovups  zmm0, zmmword ptr [rbp-0x9B0]
       vmovups  zmm1, zmmword ptr [rbp-0x270]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x4F0]
       vmovups  zmmword ptr [rbp-0x270], zmm1
       vmovups  zmm0, zmmword ptr [rbp-0x9B0]
       vmovups  zmm1, zmmword ptr [rbp-0x470]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x530]
       vmovups  zmmword ptr [rbp-0x470], zmm1
       movsxd   rax, dword ptr [rbp-0x48]
       mov      rcx, qword ptr [rbp-0x478]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x478], rax
       mov      rax, qword ptr [rbp-0x480]
       add      rax, 32
       mov      qword ptr [rbp-0x480], rax
       mov      eax, dword ptr [rbp-0x48C]
       inc      eax
       mov      dword ptr [rbp-0x48C], eax
 
G_M000_IG07:                ;; offset=0x0552
       mov      eax, dword ptr [rbp-0x9C8]
       dec      eax
       mov      dword ptr [rbp-0x9C8], eax
       cmp      dword ptr [rbp-0x9C8], 0
       jg       SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x0569
       lea      rdi, [rbp-0x9C8]
       mov      esi, 487
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG09:                ;; offset=0x057A
       mov      eax, dword ptr [rbp-0x48C]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG05
       mov      rdi, 0x7A849276190C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x48]
       add      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x9B8], rax
       mov      rax, qword ptr [rbp-0x9B8]
       add      rax, 512
       mov      qword ptr [rbp-0x488], rax
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0xB0]
       vmovups  zmmword ptr [rax], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0xF0]
       vmovups  zmmword ptr [rax+0x40], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x130]
       vmovups  zmmword ptr [rax+0x80], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x170]
       vmovups  zmmword ptr [rax+0xC0], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x1B0]
       vmovups  zmmword ptr [rax+0x100], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x1F0]
       vmovups  zmmword ptr [rax+0x140], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x230]
       vmovups  zmmword ptr [rax+0x180], zmm0
       mov      rax, qword ptr [rbp-0x9B8]
       vmovups  zmm0, zmmword ptr [rbp-0x270]
       vmovups  zmmword ptr [rax+0x1C0], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x2B0]
       vmovups  zmmword ptr [rax], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x2F0]
       vmovups  zmmword ptr [rax+0x40], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x330]
       vmovups  zmmword ptr [rax+0x80], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x370]
       vmovups  zmmword ptr [rax+0xC0], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x3B0]
       vmovups  zmmword ptr [rax+0x100], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x3F0]
 
G_M000_IG10:                ;; offset=0x0711
       vmovups  zmmword ptr [rax+0x140], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x430]
       vmovups  zmmword ptr [rax+0x180], zmm0
       mov      rax, qword ptr [rbp-0x488]
       vmovups  zmm0, zmmword ptr [rbp-0x470]
       vmovups  zmmword ptr [rax+0x1C0], zmm0
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 32
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG11:                ;; offset=0x0751
       mov      eax, dword ptr [rbp-0x9C8]
       dec      eax
       mov      dword ptr [rbp-0x9C8], eax
       cmp      dword ptr [rbp-0x9C8], 0
       jg       SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x0768
       lea      rdi, [rbp-0x9C8]
       mov      esi, 742
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG13:                ;; offset=0x0779
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 32
       cmp      eax, dword ptr [rbp-0x48]
       jle      G_M000_IG04
       jmp      G_M000_IG19
 
G_M000_IG14:                ;; offset=0x078D
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x570], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x5B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x5F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x630], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x670], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x6B0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x6F0], zmm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  zmmword ptr [rbp-0x730], zmm0
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       imul     eax, dword ptr [rbp-0x48]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       movsxd   rcx, dword ptr [rbp-0x50]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x738], rax
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x740], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x744], eax
       jmp      G_M000_IG16
 
G_M000_IG15:                ;; offset=0x0849
       mov      rdi, 0x7A8492761910
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x738]
       vmovups  zmm0, zmmword ptr [rax]
       vmovups  zmmword ptr [rbp-0x7B0], zmm0
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax]
       vmovups  zmm1, zmmword ptr [rbp-0x570]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x570], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x04]
       vmovups  zmm1, zmmword ptr [rbp-0x5B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x5B0], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x08]
       vmovups  zmm1, zmmword ptr [rbp-0x5F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x5F0], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x0C]
       vmovups  zmm1, zmmword ptr [rbp-0x630]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x630], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x10]
       vmovups  zmm1, zmmword ptr [rbp-0x670]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x670], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x14]
       vmovups  zmm1, zmmword ptr [rbp-0x6B0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x6B0], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x18]
       vmovups  zmm1, zmmword ptr [rbp-0x6F0]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x6F0], zmm1
       mov      rax, qword ptr [rbp-0x740]
       vbroadcastss zmm0, dword ptr [rax+0x1C]
       vmovups  zmm1, zmmword ptr [rbp-0x730]
       vfmadd231ps zmm1, zmm0, zmmword ptr [rbp-0x7B0]
       vmovups  zmmword ptr [rbp-0x730], zmm1
       movsxd   rax, dword ptr [rbp-0x48]
       mov      rcx, qword ptr [rbp-0x738]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x738], rax
       mov      rax, qword ptr [rbp-0x740]
       add      rax, 32
       mov      qword ptr [rbp-0x740], rax
       mov      eax, dword ptr [rbp-0x744]
       inc      eax
       mov      dword ptr [rbp-0x744], eax
 
G_M000_IG16:                ;; offset=0x0A04
       mov      eax, dword ptr [rbp-0x9C8]
       dec      eax
       mov      dword ptr [rbp-0x9C8], eax
       cmp      dword ptr [rbp-0x9C8], 0
       jg       SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x0A1B
       lea      rdi, [rbp-0x9C8]
       mov      esi, 0x42E
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG18:                ;; offset=0x0A2C
       mov      eax, dword ptr [rbp-0x744]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG15
       mov      rdi, 0x7A8492761914
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x48]
       add      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x9C0], rax
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x570]
       vmovups  zmmword ptr [rax], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x5B0]
       vmovups  zmmword ptr [rax+0x40], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x5F0]
       vmovups  zmmword ptr [rax+0x80], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x630]
       vmovups  zmmword ptr [rax+0xC0], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x670]
       vmovups  zmmword ptr [rax+0x100], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x6B0]
       vmovups  zmmword ptr [rax+0x140], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x6F0]
       vmovups  zmmword ptr [rax+0x180], zmm0
       mov      rax, qword ptr [rbp-0x9C0]
       vmovups  zmm0, zmmword ptr [rbp-0x730]
       vmovups  zmmword ptr [rax+0x1C0], zmm0
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 16
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG19:                ;; offset=0x0B30
       mov      eax, dword ptr [rbp-0x9C8]
       dec      eax
       mov      dword ptr [rbp-0x9C8], eax
       cmp      dword ptr [rbp-0x9C8], 0
       jg       SHORT G_M000_IG21
 
G_M000_IG20:                ;; offset=0x0B47
       lea      rdi, [rbp-0x9C8]
       mov      esi, 0x4B1
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG21:                ;; offset=0x0B58
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp-0x48]
       jl       G_M000_IG14
       mov      rdi, 0x7A8492761918
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       inc      eax
       mov      dword ptr [rbp-0x4C], eax
 
G_M000_IG22:                ;; offset=0x0B7B
       mov      eax, dword ptr [rbp-0x9C8]
       dec      eax
       mov      dword ptr [rbp-0x9C8], eax
       cmp      dword ptr [rbp-0x9C8], 0
       jg       SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x0B92
       lea      rdi, [rbp-0x9C8]
       mov      esi, 0x4BD
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG24:                ;; offset=0x0BA3
       cmp      dword ptr [rbp-0x4C], 16
       jl       G_M000_IG03
       mov      rdi, 0x7A849276191C
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG25:                ;; offset=0x0BBD
       vzeroupper 
       add      rsp, 0x9D0
       pop      rbp
       ret      
 
; Total bytes of code 3017

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:OutputWinograd512(ptr,ptr,int,int,int,int,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 15888

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
       jge      SHORT G_M000_IG12
 
G_M000_IG11:                ;; offset=0x024C
       mov      edi, r8d
       shl      edi, 4
       movsxd   rdi, edi
       vmovups  zmmword ptr [rax+4*rdi], zmm1
       cmp      edx, r8d
       jge      SHORT G_M000_IG12
       lea      edx, [r8+0x01]
       shl      edx, 4
       movsxd   rdx, edx
       vmovups  zmmword ptr [rax+4*rdx], zmm3
 
G_M000_IG12:                ;; offset=0x0272
       inc      r13d
       cmp      r13d, r11d
       jge      G_M000_IG05
 
G_M000_IG13:                ;; offset=0x027E
       mov      ebx, dword ptr [rbp+0x10]
       mov      rdi, qword ptr [rbp-0x30]
       jmp      G_M000_IG07
 
; Total bytes of code 650

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 20968

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     rbx
       lea      rbp, [rsp+0x20]
 
G_M000_IG02:                ;; offset=0x000D
       xor      eax, eax
       jmp      SHORT G_M000_IG04
       align    [0 bytes for IG07]
 
G_M000_IG03:                ;; offset=0x0011
       cmp      r9d, r8d
       jl       G_M000_IG10
       inc      eax
       cmp      eax, 16
       jge      G_M000_IG09
 
G_M000_IG04:                ;; offset=0x0025
       xor      r9d, r9d
       mov      r10d, eax
       imul     r10d, ecx
       mov      r11d, r10d
       imul     r11d, r8d
       movsxd   r11, r11d
       lea      r11, [rsi+4*r11]
       shl      r10d, 3
       movsxd   r10, r10d
       lea      r10, [rdi+4*r10]
 
G_M000_IG05:                ;; offset=0x0048
       lea      ebx, [r9+0x20]
       cmp      ebx, r8d
       jg       SHORT G_M000_IG03
 
G_M000_IG06:                ;; offset=0x0051
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       vxorps   ymm8, ymm8, ymm8
       vxorps   ymm9, ymm9, ymm9
       vxorps   ymm10, ymm10, ymm10
       vxorps   ymm11, ymm11, ymm11
       vxorps   ymm12, ymm12, ymm12
       vxorps   ymm13, ymm13, ymm13
       vxorps   ymm14, ymm14, ymm14
       vxorps   ymm15, ymm15, ymm15
       movsxd   rbx, r9d
       lea      rbx, [r11+4*rbx]
       mov      r15, r10
       xor      r14d, r14d
       cmp      r14d, ecx
       jge      G_M000_IG08
 
G_M000_IG07:                ;; offset=0x00AF
       vmovups  zmm16, zmmword ptr [rbx]
       vmovups  zmm17, zmmword ptr [rbx+0x40]
       vbroadcastss zmm18, dword ptr [r15]
       vfmadd231ps zmm0, zmm16, zmm18
       vfmadd231ps zmm8, zmm17, zmm18
       vbroadcastss zmm18, dword ptr [r15+0x04]
       vfmadd231ps zmm1, zmm16, zmm18
       vfmadd231ps zmm9, zmm17, zmm18
       vbroadcastss zmm18, dword ptr [r15+0x08]
       vfmadd231ps zmm2, zmm16, zmm18
       vfmadd231ps zmm10, zmm17, zmm18
       vbroadcastss zmm18, dword ptr [r15+0x0C]
       vfmadd231ps zmm3, zmm16, zmm18
       vfmadd231ps zmm11, zmm17, zmm18
       vbroadcastss zmm18, dword ptr [r15+0x10]
       vfmadd231ps zmm4, zmm16, zmm18
       vfmadd231ps zmm12, zmm17, zmm18
       vbroadcastss zmm18, dword ptr [r15+0x14]
       vfmadd231ps zmm5, zmm16, zmm18
       vfmadd231ps zmm13, zmm17, zmm18
       vbroadcastss zmm18, dword ptr [r15+0x18]
       vfmadd231ps zmm6, zmm16, zmm18
       vfmadd231ps zmm14, zmm17, zmm18
       vbroadcastss zmm18, dword ptr [r15+0x1C]
       vfmadd231ps zmm7, zmm16, zmm18
       vfmadd231ps zmm15, zmm17, zmm18
       movsxd   r13, r8d
       lea      rbx, [rbx+4*r13]
       add      r15, 32
       inc      r14d
       cmp      r14d, ecx
       jl       G_M000_IG07
 
G_M000_IG08:                ;; offset=0x016A
       mov      ebx, eax
       imul     ebx, r8d
       add      ebx, r9d
       shl      ebx, 3
       movsxd   rbx, ebx
       lea      rbx, [rdx+4*rbx]
       lea      r15, [rbx+0x200]
       vmovups  zmmword ptr [rbx], zmm0
       vmovups  zmmword ptr [rbx+0x40], zmm1
       vmovups  zmmword ptr [rbx+0x80], zmm2
       vmovups  zmmword ptr [rbx+0xC0], zmm3
       vmovups  zmmword ptr [rbx+0x100], zmm4
       vmovups  zmmword ptr [rbx+0x140], zmm5
       vmovups  zmmword ptr [rbx+0x180], zmm6
       vmovups  zmmword ptr [rbx+0x1C0], zmm7
       vmovups  zmmword ptr [r15], zmm8
       vmovups  zmmword ptr [r15+0x40], zmm9
       vmovups  zmmword ptr [r15+0x80], zmm10
       vmovups  zmmword ptr [r15+0xC0], zmm11
       vmovups  zmmword ptr [r15+0x100], zmm12
       vmovups  zmmword ptr [r15+0x140], zmm13
       vmovups  zmmword ptr [r15+0x180], zmm14
       vmovups  zmmword ptr [r15+0x1C0], zmm15
       add      r9d, 32
       jmp      G_M000_IG05
 
G_M000_IG09:                ;; offset=0x01FB
       vzeroupper 
       pop      rbx
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG10:                ;; offset=0x0207
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       movsxd   rbx, r9d
       lea      rbx, [r11+4*rbx]
       mov      r15, r10
       xor      r14d, r14d
       cmp      r14d, ecx
       jge      SHORT G_M000_IG12
 
G_M000_IG11:                ;; offset=0x0239
       vmovups  zmm8, zmmword ptr [rbx]
       vfmadd231ps zmm0, zmm8, dword ptr [r15] {1to16}
       vfmadd231ps zmm1, zmm8, dword ptr [r15+0x04] {1to16}
       vfmadd231ps zmm2, zmm8, dword ptr [r15+0x08] {1to16}
       vfmadd231ps zmm3, zmm8, dword ptr [r15+0x0C] {1to16}
       vfmadd231ps zmm4, zmm8, dword ptr [r15+0x10] {1to16}
       vfmadd231ps zmm5, zmm8, dword ptr [r15+0x14] {1to16}
       vfmadd231ps zmm6, zmm8, dword ptr [r15+0x18] {1to16}
       vfmadd231ps zmm7, zmm8, dword ptr [r15+0x1C] {1to16}
       movsxd   r13, r8d
       lea      rbx, [rbx+4*r13]
       add      r15, 32
       inc      r14d
       cmp      r14d, ecx
       jl       SHORT G_M000_IG11
 
G_M000_IG12:                ;; offset=0x0289
       mov      ebx, eax
       imul     ebx, r8d
       add      ebx, r9d
       shl      ebx, 3
       movsxd   rbx, ebx
       lea      rbx, [rdx+4*rbx]
       vmovups  zmmword ptr [rbx], zmm0
       vmovups  zmmword ptr [rbx+0x40], zmm1
       vmovups  zmmword ptr [rbx+0x80], zmm2
       vmovups  zmmword ptr [rbx+0xC0], zmm3
       vmovups  zmmword ptr [rbx+0x100], zmm4
       vmovups  zmmword ptr [rbx+0x140], zmm5
       vmovups  zmmword ptr [rbx+0x180], zmm6
       vmovups  zmmword ptr [rbx+0x1C0], zmm7
       add      r9d, 16
       jmp      G_M000_IG03
 
; Total bytes of code 732

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 320
       lea      rbp, [rsp+0x140]
       vxorps   xmm8, xmm8, xmm8
       vmovdqu32 zmmword ptr [rbp-0x130], zmm8
       vmovdqu32 zmmword ptr [rbp-0xF0], zmm8
       vmovdqu32 zmmword ptr [rbp-0xB0], zmm8
       vmovdqu32 zmmword ptr [rbp-0x80], zmm8
       xor      eax, eax
       mov      qword ptr [rbp-0x40], rax
       mov      bword ptr [rbp-0x38], rdi
       mov      qword ptr [rbp-0x30], rsi
 
G_M000_IG02:                ;; offset=0x0048
       mov      dword ptr [rbp-0x138], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x3C], eax
       lea      rdi, [rbp-0x38]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0x50], rax
       mov      rax, bword ptr [rbp-0x50]
       mov      qword ptr [rbp-0x140], rax
       mov      rax, qword ptr [rbp-0x140]
       mov      qword ptr [rbp-0x48], rax
       vbroadcastss zmm0, dword ptr [reloc @RWD00]
       vmovups  zmmword ptr [rbp-0xB0], zmm0
       vbroadcastss zmm0, dword ptr [reloc @RWD04]
       vmovups  zmmword ptr [rbp-0xF0], zmm0
       jmp      SHORT G_M000_IG06
 
G_M000_IG03:                ;; offset=0x00A5
       movsxd   rax, dword ptr [rbp-0x3C]
       mov      rcx, qword ptr [rbp-0x48]
       vmovups  zmm0, zmmword ptr [rcx+4*rax]
       vpandd   zmm0, zmm0, zmmword ptr [rbp-0xB0]
       vcmpgtps k1, zmm0, zmmword ptr [rbp-0xF0]
       kmovw    eax, k1
       test     rax, rax
       je       SHORT G_M000_IG05
       mov      rdi, 0x7A8492764CF0
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG04:                ;; offset=0x00E3
       vzeroupper 
       add      rsp, 320
       pop      rbp
       ret      
 
G_M000_IG05:                ;; offset=0x00EF
       mov      rdi, 0x7A8492764CF4
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       add      eax, 16
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG06:                ;; offset=0x0107
       mov      eax, dword ptr [rbp-0x138]
       dec      eax
       mov      dword ptr [rbp-0x138], eax
       cmp      dword ptr [rbp-0x138], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x011E
       lea      rdi, [rbp-0x138]
       mov      esi, 93
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x012F
       mov      eax, dword ptr [rbp-0x30]
       add      eax, -16
       cmp      dword ptr [rbp-0x3C], eax
       jle      G_M000_IG03
       xor      eax, eax
       mov      bword ptr [rbp-0x50], rax
       jmp      SHORT G_M000_IG12
 
G_M000_IG09:                ;; offset=0x0146
       mov      eax, dword ptr [rbp-0x30]
       cmp      dword ptr [rbp-0x3C], eax
       jae      G_M000_IG16
       mov      eax, dword ptr [rbp-0x3C]
       mov      rcx, bword ptr [rbp-0x38]
       vmovss   xmm0, dword ptr [rcx+4*rax]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       jbe      SHORT G_M000_IG11
       mov      rdi, 0x7A8492764CF8
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG10:                ;; offset=0x0181
       vzeroupper 
       add      rsp, 320
       pop      rbp
       ret      
 
G_M000_IG11:                ;; offset=0x018D
       mov      rdi, 0x7A8492764CFC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       inc      eax
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG12:                ;; offset=0x01A4
       mov      eax, dword ptr [rbp-0x138]
       dec      eax
       mov      dword ptr [rbp-0x138], eax
       cmp      dword ptr [rbp-0x138], 0
       jg       SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x01BB
       lea      rdi, [rbp-0x138]
       mov      esi, 235
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG14:                ;; offset=0x01CC
       mov      eax, dword ptr [rbp-0x3C]
       cmp      eax, dword ptr [rbp-0x30]
       jl       G_M000_IG09
       mov      rdi, 0x7A8492764D00
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, 1
 
G_M000_IG15:                ;; offset=0x01EC
       vzeroupper 
       add      rsp, 320
       pop      rbp
       ret      
 
G_M000_IG16:                ;; offset=0x01F8
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	7E7FFFFFh		; 8.50706e+37
RWD08  	dd	00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 510

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x5d
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 57
; 1 inlinees with PGO data; 0 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       mov      rbp, rsp
       mov      eax, dword ptr [rbp+0x114]
       mov      rcx, qword ptr [rbp+0x108]
       vmovups  zmm0, zmmword ptr [rbp+0xA0]
       vmovups  zmm1, zmmword ptr [rbp+0x60]
 
G_M000_IG02:                ;; offset=0x0029
       mov      edx, dword ptr [rbp+0x120]
       lea      edi, [rdx-0x10]
       cmp      eax, edi
       jg       SHORT G_M000_IG05
       align    [0 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0036
       movsxd   rsi, eax
       vpandd   zmm2, zmm0, zmmword ptr [rcx+4*rsi]
       vcmpgtps k1, zmm2, zmm1
       kmovw    esi, k1
       test     rsi, rsi
       jne      SHORT G_M000_IG11
 
G_M000_IG04:                ;; offset=0x0050
       add      eax, 16
       cmp      eax, edi
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x0057
       xor      ecx, ecx
       mov      bword ptr [rbp+0x100], rcx
       cmp      eax, edx
       jl       SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0064
       mov      eax, 1
 
G_M000_IG07:                ;; offset=0x0069
       vzeroupper 
       add      rsp, 336
       pop      rbp
       ret      
 
G_M000_IG08:                ;; offset=0x0075
       test     eax, eax
       jl       SHORT G_M000_IG13
 
G_M000_IG09:                ;; offset=0x0079
       mov      rcx, bword ptr [rbp+0x118]
       mov      edi, eax
       vmovss   xmm0, dword ptr [rcx+4*rdi]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD00]
       vucomiss xmm0, dword ptr [reloc @RWD16]
       ja       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x0099
       inc      eax
       cmp      eax, edx
       jl       SHORT G_M000_IG09
       jmp      SHORT G_M000_IG06
 
G_M000_IG11:                ;; offset=0x00A1
       xor      eax, eax
 
G_M000_IG12:                ;; offset=0x00A3
       vzeroupper 
       add      rsp, 336
       pop      rbp
       ret      
 
G_M000_IG13:                ;; offset=0x00AF
       cmp      eax, edx
       jae      SHORT G_M000_IG15
       mov      rcx, bword ptr [rbp+0x118]
       mov      edi, eax
       vmovss   xmm0, dword ptr [rcx+4*rdi]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD00]
       vucomiss xmm0, dword ptr [reloc @RWD16]
       ja       SHORT G_M000_IG11
 
G_M000_IG14:                ;; offset=0x00D3
       inc      eax
       cmp      eax, edx
       jl       SHORT G_M000_IG13
       jmp      SHORT G_M000_IG06
 
G_M000_IG15:                ;; offset=0x00DB
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh
RWD16  	dd	7E7FFFFFh		; 8.50706e+37

; Total bytes of code 225

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 80
       lea      rbp, [rsp+0x50]
       vxorps   xmm8, xmm8, xmm8
       vmovdqu  ymmword ptr [rbp-0x50], ymm8
       vmovdqa  xmmword ptr [rbp-0x30], xmm8
       mov      dword ptr [rbp-0x04], edi
       mov      dword ptr [rbp-0x08], esi
       mov      dword ptr [rbp-0x0C], edx
       mov      dword ptr [rbp-0x10], ecx
       mov      bword ptr [rbp-0x18], r8
       mov      bword ptr [rbp-0x20], r9
 
G_M000_IG02:                ;; offset=0x002D
       xor      eax, eax
       mov      dword ptr [rbp-0x24], eax
       mov      rax, bword ptr [rbp+0x10]
       xor      ecx, ecx
       mov      dword ptr [rax], ecx
       mov      rax, bword ptr [rbp-0x20]
       mov      ecx, dword ptr [rbp-0x24]
       mov      dword ptr [rax], ecx
       mov      rax, bword ptr [rbp-0x18]
       mov      ecx, dword ptr [rbp-0x24]
       mov      dword ptr [rax], ecx
       cmp      dword ptr [rbp-0x04], 0
       jle      SHORT G_M000_IG04
       cmp      dword ptr [rbp-0x08], 0
       jle      SHORT G_M000_IG07
       cmp      dword ptr [rbp-0x0C], 0
       jle      SHORT G_M000_IG06
       cmp      dword ptr [rbp-0x10], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG03:                ;; offset=0x0064
       mov      rdi, 0x7A84928E1610
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG04:                ;; offset=0x0073
       mov      rdi, 0x7A84928E1614
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG05:                ;; offset=0x0084
       add      rsp, 80
       pop      rbp
       ret      
 
G_M000_IG06:                ;; offset=0x008A
       mov      rdi, 0x7A84928E1618
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG04
 
G_M000_IG07:                ;; offset=0x009B
       mov      rdi, 0x7A84928E161C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG04
 
G_M000_IG08:                ;; offset=0x00AC
       movsxd   rax, dword ptr [rbp-0x04]
       imul     rax, rax, 16
       jo       SHORT G_M000_IG09
       imul     rax, rax, 8
       jo       SHORT G_M000_IG09
       mov      qword ptr [rbp-0x30], rax
       movsxd   rax, dword ptr [rbp-0x08]
       imul     rax, rax, 16
       jo       SHORT G_M000_IG09
       imul     rax, rax, 8
       jo       SHORT G_M000_IG09
       mov      qword ptr [rbp-0x38], rax
       movsxd   rax, dword ptr [rbp-0x08]
       movsxd   rcx, dword ptr [rbp-0x0C]
       imul     rax, rcx
       jo       SHORT G_M000_IG09
       movsxd   rcx, dword ptr [rbp-0x10]
       imul     rax, rcx
       jo       SHORT G_M000_IG09
       mov      qword ptr [rbp-0x40], rax
       mov      rax, qword ptr [rbp-0x30]
       add      rax, qword ptr [rbp-0x38]
       jo       SHORT G_M000_IG09
       add      rax, qword ptr [rbp-0x40]
       jo       SHORT G_M000_IG09
       cmp      rax, 0x1000000
       jle      SHORT G_M000_IG10
       xor      eax, eax
       mov      dword ptr [rbp-0x44], eax
       jmp      SHORT G_M000_IG11
 
G_M000_IG09:                ;; offset=0x010F
       call     CORINFO_HELP_OVERFLOW
       int3     
 
G_M000_IG10:                ;; offset=0x0115
       mov      rdi, 0x7A84928E1620
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, bword ptr [rbp-0x18]
       mov      ecx, dword ptr [rbp-0x30]
       mov      dword ptr [rax], ecx
       mov      rax, bword ptr [rbp-0x20]
       mov      ecx, dword ptr [rbp-0x38]
       mov      dword ptr [rax], ecx
       mov      rax, bword ptr [rbp+0x10]
       mov      ecx, dword ptr [rbp-0x40]
       mov      dword ptr [rax], ecx
       mov      dword ptr [rbp-0x44], 1
 
G_M000_IG11:                ;; offset=0x0146
       mov      rdi, 0x7A84928E1628
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
 
G_M000_IG12:                ;; offset=0x0158
       add      rsp, 80
       pop      rbp
       ret      
 
G_M000_IG13:                ;; offset=0x015E
       push     rax
 
G_M000_IG14:                ;; offset=0x015F
       mov      gword ptr [rbp-0x50], rdi
       mov      rdi, 0x7A84928E1624
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
       mov      dword ptr [rbp-0x44], eax
       lea      rax, G_M000_IG11
 
G_M000_IG15:                ;; offset=0x017E
       add      rsp, 8
       ret      
 
; Total bytes of code 387

; Assembly listing for method KernelAccess:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; partially interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 64
       lea      rbp, [rsp+0x40]
       xor      eax, eax
       mov      qword ptr [rbp-0x28], rax
       mov      qword ptr [rbp-0x30], rax
       mov      dword ptr [rbp-0x04], edi
       mov      dword ptr [rbp-0x08], esi
       mov      dword ptr [rbp-0x0C], edx
       mov      dword ptr [rbp-0x10], ecx
       mov      bword ptr [rbp-0x18], r8
       mov      bword ptr [rbp-0x20], r9
 
G_M000_IG02:                ;; offset=0x0028
       mov      rax, 0x7A7CAD800198
       mov      rax, gword ptr [rax]
       mov      gword ptr [rbp-0x28], rax
       mov      rdi, gword ptr [rbp-0x28]
       mov      rsi, 0x7A84928E6CF8
       call     CORINFO_HELP_DELEGATEPROFILE32
       mov      rax, gword ptr [rbp-0x28]
       mov      gword ptr [rbp-0x30], rax
       mov      rax, bword ptr [rbp-0x20]
       mov      bword ptr [rsp], rax
       mov      rax, bword ptr [rbp+0x10]
       mov      bword ptr [rsp+0x08], rax
       mov      rax, gword ptr [rbp-0x30]
       mov      esi, dword ptr [rbp-0x04]
       mov      edx, dword ptr [rbp-0x08]
       mov      ecx, dword ptr [rbp-0x0C]
       mov      r8d, dword ptr [rbp-0x10]
       mov      r9, bword ptr [rbp-0x18]
       mov      rdi, gword ptr [rax+0x08]
       mov      rax, gword ptr [rbp-0x30]
       call     [rax+0x18]KernelAccess+PlanCall:Invoke(int,int,int,int,byref,byref,byref):bool:this
       nop      
 
G_M000_IG03:                ;; offset=0x0086
       add      rsp, 64
       pop      rbp
       ret      
 
; Total bytes of code 140

; Assembly listing for method KernelAccess:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; partially interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 208
       lea      rbp, [rsp+0xD0]
       xor      eax, eax
       mov      qword ptr [rbp-0x38], rax
       mov      qword ptr [rbp-0x40], rax
       mov      bword ptr [rbp-0x10], rdi
       mov      qword ptr [rbp-0x08], rsi
       mov      bword ptr [rbp-0x20], rdx
       mov      qword ptr [rbp-0x18], rcx
       mov      bword ptr [rbp-0x30], r8
       mov      qword ptr [rbp-0x28], r9
 
G_M000_IG02:                ;; offset=0x0032
       mov      rax, 0x7A7CAD8001A8
       mov      rax, gword ptr [rax]
       mov      gword ptr [rbp-0x38], rax
       mov      rdi, gword ptr [rbp-0x38]
       mov      rsi, 0x7A84928E6E30
       call     CORINFO_HELP_DELEGATEPROFILE32
       mov      rax, gword ptr [rbp-0x38]
       mov      gword ptr [rbp-0x40], rax
       lea      rdi, [rsp]
       lea      rsi, [rbp-0x30]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       lea      rdi, [rsp+0x10]
       lea      rsi, [rbp+0x10]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp+0x10], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       lea      rdi, [rsp+0x20]
       lea      rsi, [rbp+0x20]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp+0x20], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       lea      rdi, [rsp+0x30]
       lea      rsi, [rbp+0x30]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp+0x30], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       lea      rdi, [rsp+0x40]
       lea      rsi, [rbp+0x40]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp+0x40], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       lea      rdi, [rsp+0x50]
       lea      rsi, [rbp+0x50]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp+0x50], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       mov      eax, dword ptr [rbp+0x68]
       mov      dword ptr [rsp+0x60], eax
       mov      eax, dword ptr [rbp+0x70]
       mov      dword ptr [rsp+0x68], eax
       mov      eax, dword ptr [rbp+0x78]
       mov      dword ptr [rsp+0x70], eax
       mov      eax, dword ptr [rbp+0x80]
       mov      dword ptr [rsp+0x78], eax
       movzx    rax, byte  ptr [rbp+0x88]
       mov      dword ptr [rsp+0x80], eax
       mov      rax, gword ptr [rbp-0x40]
       mov      rsi, bword ptr [rbp-0x10]
 
G_M000_IG03:                ;; offset=0x0133
       mov      rdx, qword ptr [rbp-0x08]
       mov      rcx, bword ptr [rbp-0x20]
       mov      r8, qword ptr [rbp-0x18]
       mov      r9d, dword ptr [rbp+0x60]
       mov      rdi, gword ptr [rax+0x08]
       mov      rax, gword ptr [rbp-0x40]
       call     [rax+0x18]KernelAccess+WinogradCall:Invoke(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool:this
       nop      
 
G_M000_IG04:                ;; offset=0x014F
       add      rsp, 208
       pop      rbp
       ret      
 
; Total bytes of code 344

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; partially interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x480
       lea      rbp, [rsp+0x480]
       xor      eax, eax
       mov      qword ptr [rbp-0x428], rax
       vxorps   xmm8, xmm8, xmm8
       mov      rax, -960
       vmovdqa  xmmword ptr [rbp+rax-0x60], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x50], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x60], rax
       mov      bword ptr [rbp-0x38], rdi
       mov      qword ptr [rbp-0x30], rsi
       mov      bword ptr [rbp-0x48], rdx
       mov      qword ptr [rbp-0x40], rcx
       mov      bword ptr [rbp-0x58], r8
       mov      qword ptr [rbp-0x50], r9
 
G_M000_IG02:                ;; offset=0x005C
       mov      dword ptr [rbp-0x418], 0x3E8
       mov      edi, dword ptr [rbp+0x60]
       mov      esi, dword ptr [rbp+0x68]
       mov      edx, dword ptr [rbp+0x70]
       mov      ecx, dword ptr [rbp+0x78]
       mov      r9d, dword ptr [rbp+0x80]
       mov      r8d, 1
       call     [Lokad.Onnx.ConvBlockedSpatial:Geometry(int,int,int,int,int,int)]
       lea      rax, [rbp-0x70]
       mov      qword ptr [rsp], rax
       lea      r9, [rbp-0x68]
       lea      r8, [rbp-0x60]
       mov      edi, dword ptr [rbp+0x60]
       mov      esi, dword ptr [rbp+0x68]
       mov      edx, dword ptr [rbp+0x70]
       mov      ecx, dword ptr [rbp+0x78]
       call     [Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool]
       test     eax, eax
       jne      SHORT G_M000_IG03
       mov      rdi, 0x7A8492761790
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG96
 
G_M000_IG03:                ;; offset=0x00BF
       mov      eax, dword ptr [rbp+0x60]
       imul     eax, dword ptr [rbp+0x70]
       jo       G_M000_IG102
       imul     eax, dword ptr [rbp+0x78]
       jo       G_M000_IG102
       cmp      dword ptr [rbp-0x30], eax
       jne      G_M000_IG07
       imul     eax, dword ptr [rbp+0x60], 16
       jo       G_M000_IG102
       imul     eax, dword ptr [rbp+0x68]
       jo       G_M000_IG102
       cmp      dword ptr [rbp-0x40], eax
       jne      G_M000_IG13
       mov      eax, dword ptr [rbp+0x28]
       cmp      eax, dword ptr [rbp-0x70]
       jne      G_M000_IG12
       cmp      dword ptr [rbp-0x50], 0
       je       SHORT G_M000_IG04
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp+0x68]
       jne      G_M000_IG11
       mov      rdi, 0x7A8492761794
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG04:                ;; offset=0x0129
       cmp      dword ptr [rbp+0x18], 0
       je       SHORT G_M000_IG05
       mov      eax, dword ptr [rbp+0x18]
       cmp      eax, dword ptr [rbp-0x70]
       jne      G_M000_IG10
       mov      rdi, 0x7A8492761798
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG05:                ;; offset=0x014A
       mov      eax, dword ptr [rbp+0x38]
       cmp      eax, dword ptr [rbp-0x60]
       jl       G_M000_IG09
       mov      eax, dword ptr [rbp+0x48]
       cmp      eax, dword ptr [rbp-0x68]
       jl       SHORT G_M000_IG08
       mov      eax, dword ptr [rbp+0x58]
       cmp      eax, dword ptr [rbp-0x70]
       jge      G_M000_IG14
 
G_M000_IG06:                ;; offset=0x016A
       mov      rdi, 0x7A849276179C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG07:                ;; offset=0x0179
       mov      rdi, 0x7A8491C8CD28
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0xC8], rax
       mov      edi, 0x102B4
       mov      rsi, 0x7A84924B9F30
       call     [CORINFO_HELP_STRCNS]
       mov      gword ptr [rbp-0x420], rax
       mov      rsi, gword ptr [rbp-0x420]
       mov      rdi, gword ptr [rbp-0xC8]
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, gword ptr [rbp-0xC8]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG08:                ;; offset=0x01CC
       mov      rdi, 0x7A84927617A0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG07
 
G_M000_IG09:                ;; offset=0x01DD
       mov      rdi, 0x7A84927617A4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG07
 
G_M000_IG10:                ;; offset=0x01EE
       mov      rdi, 0x7A84927617A8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG11:                ;; offset=0x0202
       mov      rdi, 0x7A84927617AC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG12:                ;; offset=0x0216
       mov      rdi, 0x7A84927617B0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG13:                ;; offset=0x022A
       mov      rdi, 0x7A84927617B4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG14:                ;; offset=0x023E
       lea      rdi, [rbp+0x30]
       mov      edx, dword ptr [rbp-0x60]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xD8], rax
       mov      qword ptr [rbp-0xD0], rdx
 
G_M000_IG15:                ;; offset=0x025B
       vmovdqu  xmm0, xmmword ptr [rbp-0xD8]
       vmovdqu  xmmword ptr [rbp+0x30], xmm0
 
G_M000_IG16:                ;; offset=0x0268
       lea      rdi, [rbp+0x40]
       mov      edx, dword ptr [rbp-0x68]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xE8], rax
       mov      qword ptr [rbp-0xE0], rdx
 
G_M000_IG17:                ;; offset=0x0285
       vmovdqu  xmm0, xmmword ptr [rbp-0xE8]
       vmovdqu  xmmword ptr [rbp+0x40], xmm0
 
G_M000_IG18:                ;; offset=0x0292
       lea      rdi, [rbp+0x50]
       mov      edx, dword ptr [rbp-0x70]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xF8], rax
       mov      qword ptr [rbp-0xF0], rdx
 
G_M000_IG19:                ;; offset=0x02AF
       vmovdqu  xmm0, xmmword ptr [rbp-0xF8]
       vmovdqu  xmmword ptr [rbp+0x50], xmm0
 
G_M000_IG20:                ;; offset=0x02BC
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu  xmmword ptr [rbp-0x118], xmm0
 
G_M000_IG21:                ;; offset=0x02C9
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x108], rax
       mov      qword ptr [rbp-0x100], rdx
       mov      rdx, bword ptr [rbp-0x108]
       mov      rcx, qword ptr [rbp-0x100]
       mov      rdi, bword ptr [rbp-0x118]
       mov      rsi, qword ptr [rbp-0x110]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG55
 
G_M000_IG22:                ;; offset=0x030F
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu32 xmmword ptr [rbp-0x140], xmm0
 
G_M000_IG23:                ;; offset=0x031B
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x130], rax
       mov      qword ptr [rbp-0x128], rdx
       mov      rdx, bword ptr [rbp-0x130]
       mov      rcx, qword ptr [rbp-0x128]
       mov      rdi, bword ptr [rbp-0x140]
       mov      rsi, qword ptr [rbp-0x138]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG75
 
G_M000_IG24:                ;; offset=0x0361
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu32 xmmword ptr [rbp-0x160], xmm0
 
G_M000_IG25:                ;; offset=0x036D
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x150], rax
       mov      qword ptr [rbp-0x148], rdx
       mov      rdx, bword ptr [rbp-0x150]
       mov      rcx, qword ptr [rbp-0x148]
       mov      rdi, bword ptr [rbp-0x160]
       mov      rsi, qword ptr [rbp-0x158]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG74
 
G_M000_IG26:                ;; offset=0x03B3
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu32 xmmword ptr [rbp-0x180], xmm0
 
G_M000_IG27:                ;; offset=0x03BF
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x170], rax
       mov      qword ptr [rbp-0x168], rdx
       mov      rdx, bword ptr [rbp-0x170]
       mov      rcx, qword ptr [rbp-0x168]
       mov      rdi, bword ptr [rbp-0x180]
       mov      rsi, qword ptr [rbp-0x178]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG73
 
G_M000_IG28:                ;; offset=0x0405
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu32 xmmword ptr [rbp-0x1A0], xmm0
 
G_M000_IG29:                ;; offset=0x0411
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x190], rax
       mov      qword ptr [rbp-0x188], rdx
       mov      rdx, bword ptr [rbp-0x190]
       mov      rcx, qword ptr [rbp-0x188]
       mov      rdi, bword ptr [rbp-0x1A0]
       mov      rsi, qword ptr [rbp-0x198]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG72
 
G_M000_IG30:                ;; offset=0x0457
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu32 xmmword ptr [rbp-0x1C0], xmm0
 
G_M000_IG31:                ;; offset=0x0463
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x1B0], rax
       mov      qword ptr [rbp-0x1A8], rdx
       mov      rdx, bword ptr [rbp-0x1B0]
       mov      rcx, qword ptr [rbp-0x1A8]
       mov      rdi, bword ptr [rbp-0x1C0]
       mov      rsi, qword ptr [rbp-0x1B8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG71
 
G_M000_IG32:                ;; offset=0x04A9
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu32 xmmword ptr [rbp-0x1E0], xmm0
 
G_M000_IG33:                ;; offset=0x04B5
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x1D0], rax
       mov      qword ptr [rbp-0x1C8], rdx
       mov      rdx, bword ptr [rbp-0x1D0]
       mov      rcx, qword ptr [rbp-0x1C8]
       mov      rdi, bword ptr [rbp-0x1E0]
       mov      rsi, qword ptr [rbp-0x1D8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG70
 
G_M000_IG34:                ;; offset=0x04FB
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu32 xmmword ptr [rbp-0x200], xmm0
 
G_M000_IG35:                ;; offset=0x0507
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x1F0], rax
       mov      qword ptr [rbp-0x1E8], rdx
       mov      rdx, bword ptr [rbp-0x1F0]
       mov      rcx, qword ptr [rbp-0x1E8]
       mov      rdi, bword ptr [rbp-0x200]
       mov      rsi, qword ptr [rbp-0x1F8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG69
 
G_M000_IG36:                ;; offset=0x054D
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu32 xmmword ptr [rbp-0x220], xmm0
 
G_M000_IG37:                ;; offset=0x0559
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x210], rax
       mov      qword ptr [rbp-0x208], rdx
       mov      rdx, bword ptr [rbp-0x210]
       mov      rcx, qword ptr [rbp-0x208]
       mov      rdi, bword ptr [rbp-0x220]
       mov      rsi, qword ptr [rbp-0x218]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG68
 
G_M000_IG38:                ;; offset=0x059F
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu32 xmmword ptr [rbp-0x240], xmm0
 
G_M000_IG39:                ;; offset=0x05AB
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x230], rax
       mov      qword ptr [rbp-0x228], rdx
       mov      rdx, bword ptr [rbp-0x230]
       mov      rcx, qword ptr [rbp-0x228]
       mov      rdi, bword ptr [rbp-0x240]
       mov      rsi, qword ptr [rbp-0x238]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG67
 
G_M000_IG40:                ;; offset=0x05F1
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu32 xmmword ptr [rbp-0x260], xmm0
 
G_M000_IG41:                ;; offset=0x05FD
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x250], rax
       mov      qword ptr [rbp-0x248], rdx
       mov      rdx, bword ptr [rbp-0x250]
       mov      rcx, qword ptr [rbp-0x248]
       mov      rdi, bword ptr [rbp-0x260]
       mov      rsi, qword ptr [rbp-0x258]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG66
 
G_M000_IG42:                ;; offset=0x0643
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu32 xmmword ptr [rbp-0x280], xmm0
 
G_M000_IG43:                ;; offset=0x064F
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x270], rax
       mov      qword ptr [rbp-0x268], rdx
       mov      rdx, bword ptr [rbp-0x270]
       mov      rcx, qword ptr [rbp-0x268]
       mov      rdi, bword ptr [rbp-0x280]
       mov      rsi, qword ptr [rbp-0x278]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG65
 
G_M000_IG44:                ;; offset=0x0695
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu32 xmmword ptr [rbp-0x2A0], xmm0
 
G_M000_IG45:                ;; offset=0x06A1
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x290], rax
       mov      qword ptr [rbp-0x288], rdx
       mov      rdx, bword ptr [rbp-0x290]
       mov      rcx, qword ptr [rbp-0x288]
       mov      rdi, bword ptr [rbp-0x2A0]
       mov      rsi, qword ptr [rbp-0x298]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG64
 
G_M000_IG46:                ;; offset=0x06E7
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu32 xmmword ptr [rbp-0x2C0], xmm0
 
G_M000_IG47:                ;; offset=0x06F3
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x2B0], rax
       mov      qword ptr [rbp-0x2A8], rdx
       mov      rdx, bword ptr [rbp-0x2B0]
       mov      rcx, qword ptr [rbp-0x2A8]
       mov      rdi, bword ptr [rbp-0x2C0]
       mov      rsi, qword ptr [rbp-0x2B8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG63
 
G_M000_IG48:                ;; offset=0x0739
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu32 xmmword ptr [rbp-0x2E0], xmm0
 
G_M000_IG49:                ;; offset=0x0745
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x2D0], rax
       mov      qword ptr [rbp-0x2C8], rdx
       mov      rdx, bword ptr [rbp-0x2D0]
       mov      rcx, qword ptr [rbp-0x2C8]
       mov      rdi, bword ptr [rbp-0x2E0]
       mov      rsi, qword ptr [rbp-0x2D8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG62
 
G_M000_IG50:                ;; offset=0x078B
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu32 xmmword ptr [rbp-0x300], xmm0
 
G_M000_IG51:                ;; offset=0x0797
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x2F0], rax
       mov      qword ptr [rbp-0x2E8], rdx
       mov      rdx, bword ptr [rbp-0x2F0]
       mov      rcx, qword ptr [rbp-0x2E8]
       mov      rdi, bword ptr [rbp-0x300]
       mov      rsi, qword ptr [rbp-0x2F8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG61
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x310], rax
       mov      qword ptr [rbp-0x308], rdx
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x320], rax
       mov      qword ptr [rbp-0x318], rdx
       mov      rdx, bword ptr [rbp-0x320]
       mov      rcx, qword ptr [rbp-0x318]
       mov      rdi, bword ptr [rbp-0x310]
       mov      rsi, qword ptr [rbp-0x308]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG60
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x330], rax
       mov      qword ptr [rbp-0x328], rdx
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x340], rax
       mov      qword ptr [rbp-0x338], rdx
       mov      rdx, bword ptr [rbp-0x340]
       mov      rcx, qword ptr [rbp-0x338]
       mov      rdi, bword ptr [rbp-0x330]
       mov      rsi, qword ptr [rbp-0x328]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG59
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x350], rax
       mov      qword ptr [rbp-0x348], rdx
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x360], rax
       mov      qword ptr [rbp-0x358], rdx
 
G_M000_IG52:                ;; offset=0x08D9
       mov      rdx, bword ptr [rbp-0x360]
       mov      rcx, qword ptr [rbp-0x358]
       mov      rdi, bword ptr [rbp-0x350]
       mov      rsi, qword ptr [rbp-0x348]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG58
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x370], rax
       mov      qword ptr [rbp-0x368], rdx
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x380], rax
       mov      qword ptr [rbp-0x378], rdx
       mov      rdx, bword ptr [rbp-0x380]
       mov      rcx, qword ptr [rbp-0x378]
       mov      rdi, bword ptr [rbp-0x370]
       mov      rsi, qword ptr [rbp-0x368]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG57
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x390], rax
       mov      qword ptr [rbp-0x388], rdx
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3A0], rax
       mov      qword ptr [rbp-0x398], rdx
       mov      rdx, bword ptr [rbp-0x3A0]
       mov      rcx, qword ptr [rbp-0x398]
       mov      rdi, bword ptr [rbp-0x390]
       mov      rsi, qword ptr [rbp-0x388]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      G_M000_IG56
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3B0], rax
       mov      qword ptr [rbp-0x3A8], rdx
       mov      rdi, bword ptr [rbp+0x20]
       mov      rsi, qword ptr [rbp+0x28]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3C0], rax
       mov      qword ptr [rbp-0x3B8], rdx
       mov      rdx, bword ptr [rbp-0x3C0]
       mov      rcx, qword ptr [rbp-0x3B8]
       mov      rdi, bword ptr [rbp-0x3B0]
       mov      rsi, qword ptr [rbp-0x3A8]
       call     [System.MemoryExtensions:Overlaps[float](System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float]):bool]
 
G_M000_IG53:                ;; offset=0x0A21
       test     eax, eax
       je       G_M000_IG76
 
G_M000_IG54:                ;; offset=0x0A29
       mov      rdi, 0x7A84927617B8
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG55:                ;; offset=0x0A38
       mov      rdi, 0x7A8491C8CD28
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0x120], rax
       mov      edi, 0x102E2
       mov      rsi, 0x7A84924B9F30
       call     [CORINFO_HELP_STRCNS]
       mov      gword ptr [rbp-0x428], rax
       mov      rsi, gword ptr [rbp-0x428]
       mov      rdi, gword ptr [rbp-0x120]
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, gword ptr [rbp-0x120]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG56:                ;; offset=0x0A8B
       mov      rdi, 0x7A84927617BC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG55
 
G_M000_IG57:                ;; offset=0x0A9C
       mov      rdi, 0x7A84927617C0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG55
 
G_M000_IG58:                ;; offset=0x0AAD
       mov      rdi, 0x7A84927617C4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG59:                ;; offset=0x0AC1
       mov      rdi, 0x7A84927617C8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG60:                ;; offset=0x0AD5
       mov      rdi, 0x7A84927617CC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG61:                ;; offset=0x0AE9
       mov      rdi, 0x7A84927617D0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG62:                ;; offset=0x0AFD
       mov      rdi, 0x7A84927617D4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG63:                ;; offset=0x0B11
       mov      rdi, 0x7A84927617D8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG64:                ;; offset=0x0B25
       mov      rdi, 0x7A84927617DC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG65:                ;; offset=0x0B39
       mov      rdi, 0x7A84927617E0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG66:                ;; offset=0x0B4D
       mov      rdi, 0x7A84927617E4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG67:                ;; offset=0x0B61
       mov      rdi, 0x7A84927617E8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG68:                ;; offset=0x0B75
       mov      rdi, 0x7A84927617EC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG69:                ;; offset=0x0B89
       mov      rdi, 0x7A84927617F0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG70:                ;; offset=0x0B9D
       mov      rdi, 0x7A84927617F4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG71:                ;; offset=0x0BB1
       mov      rdi, 0x7A84927617F8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG72:                ;; offset=0x0BC5
       mov      rdi, 0x7A84927617FC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG73:                ;; offset=0x0BD9
       mov      rdi, 0x7A8492761800
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG74:                ;; offset=0x0BED
       mov      rdi, 0x7A8492761804
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG75:                ;; offset=0x0C01
       mov      rdi, 0x7A8492761808
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG76:                ;; offset=0x0C15
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG77
       mov      rdi, 0x7A849276180C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG77:                ;; offset=0x0C2D
       cmp      dword ptr [rbp+0x80], 8
       jne      SHORT G_M000_IG78
       mov      rdi, 0x7A8492761810
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG78:                ;; offset=0x0C45
       mov      rdi, bword ptr [rbp-0x38]
       mov      rsi, qword ptr [rbp-0x30]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG80
       mov      rdi, bword ptr [rbp-0x48]
       mov      rsi, qword ptr [rbp-0x40]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG82
       mov      rdi, bword ptr [rbp-0x58]
       mov      rsi, qword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG81
       mov      rdi, bword ptr [rbp+0x10]
       mov      rsi, qword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG83
 
G_M000_IG79:                ;; offset=0x0C8D
       mov      rdi, 0x7A8492761814
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG80:                ;; offset=0x0C9C
       mov      rdi, 0x7A8492761818
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG96
 
G_M000_IG81:                ;; offset=0x0CB0
       mov      rdi, 0x7A849276181C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG80
 
G_M000_IG82:                ;; offset=0x0CC1
       mov      rdi, 0x7A8492761820
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG80
 
G_M000_IG83:                ;; offset=0x0CD2
       mov      eax, dword ptr [rbp+0x78]
       inc      eax
       mov      dword ptr [rbp-0x44C], eax
       mov      eax, dword ptr [rbp-0x44C]
       shr      eax, 31
       add      eax, dword ptr [rbp-0x44C]
       sar      eax, 1
       mov      dword ptr [rbp-0x74], eax
       mov      eax, dword ptr [rbp+0x70]
       add      eax, 1
       jo       G_M000_IG102
       mov      dword ptr [rbp-0x450], eax
       mov      eax, dword ptr [rbp-0x450]
       shr      eax, 31
       add      eax, dword ptr [rbp-0x450]
       sar      eax, 1
       imul     eax, dword ptr [rbp-0x74]
       jo       G_M000_IG102
       mov      dword ptr [rbp-0x78], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x7C], eax
       jmp      G_M000_IG91
 
G_M000_IG84:                ;; offset=0x0D2B
       mov      eax, dword ptr [rbp-0x78]
       mov      esi, eax
       sub      esi, dword ptr [rbp-0x7C]
       mov      edi, 8
       call     [System.Math:Min(int,int):int]
       mov      dword ptr [rbp-0x80], eax
       mov      eax, dword ptr [rbp+0x78]
       mov      dword ptr [rsp], eax
       mov      eax, dword ptr [rbp-0x74]
       mov      dword ptr [rsp+0x08], eax
       mov      eax, dword ptr [rbp-0x7C]
       mov      dword ptr [rsp+0x10], eax
       mov      eax, dword ptr [rbp-0x80]
       mov      dword ptr [rsp+0x18], eax
       mov      rdx, bword ptr [rbp+0x30]
       mov      rcx, qword ptr [rbp+0x38]
       mov      rdi, bword ptr [rbp-0x38]
       mov      rsi, qword ptr [rbp-0x30]
       mov      r8d, dword ptr [rbp+0x60]
       mov      r9d, dword ptr [rbp+0x70]
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      rdi, bword ptr [rbp+0x30]
       mov      rsi, qword ptr [rbp+0x38]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3D0], rax
       mov      qword ptr [rbp-0x3C8], rdx
       mov      rdi, bword ptr [rbp-0x3D0]
       mov      rsi, qword ptr [rbp-0x3C8]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG85
       mov      rdi, 0x7A8492761824
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG96
 
G_M000_IG85:                ;; offset=0x0DC2
       lea      rdi, [rbp+0x30]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xA8], rax
       mov      rax, bword ptr [rbp-0xA8]
       mov      qword ptr [rbp-0x430], rax
       mov      rax, qword ptr [rbp-0x430]
       mov      qword ptr [rbp-0x88], rax
       lea      rdi, [rbp-0x48]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xB0], rax
       mov      rax, bword ptr [rbp-0xB0]
       mov      qword ptr [rbp-0x438], rax
       mov      rax, qword ptr [rbp-0x438]
       mov      qword ptr [rbp-0x90], rax
       lea      rdi, [rbp+0x40]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xB8], rax
       mov      rax, bword ptr [rbp-0xB8]
       mov      qword ptr [rbp-0x440], rax
       mov      rax, qword ptr [rbp-0x440]
       mov      qword ptr [rbp-0x98], rax
       lea      rdi, [rbp+0x50]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xC0], rax
       mov      rax, bword ptr [rbp-0xC0]
       mov      qword ptr [rbp-0x448], rax
       mov      rax, qword ptr [rbp-0x448]
       mov      qword ptr [rbp-0xA0], rax
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG86
       mov      rdi, qword ptr [rbp-0x88]
       mov      rsi, qword ptr [rbp-0x90]
       mov      rdx, qword ptr [rbp-0x98]
       mov      ecx, dword ptr [rbp+0x60]
       mov      r8d, dword ptr [rbp+0x68]
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int)]
       jmp      SHORT G_M000_IG87
 
G_M000_IG86:                ;; offset=0x0EA3
       mov      rdi, 0x7A8492761828
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rdi, qword ptr [rbp-0x88]
       mov      rsi, qword ptr [rbp-0x90]
       mov      rdx, qword ptr [rbp-0x98]
       mov      ecx, dword ptr [rbp+0x60]
       mov      r8d, dword ptr [rbp+0x68]
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int)]
 
G_M000_IG87:                ;; offset=0x0ED4
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3E0], rax
       mov      qword ptr [rbp-0x3D8], rdx
       mov      rdi, bword ptr [rbp-0x3E0]
       mov      rsi, qword ptr [rbp-0x3D8]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG88
       mov      rdi, 0x7A849276182C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG96
 
G_M000_IG88:                ;; offset=0x0F1C
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG89
       mov      eax, dword ptr [rbp-0x7C]
       mov      dword ptr [rsp], eax
       mov      eax, dword ptr [rbp-0x80]
       mov      dword ptr [rsp+0x08], eax
       mov      rdi, qword ptr [rbp-0x98]
       mov      rsi, qword ptr [rbp-0xA0]
       mov      edx, dword ptr [rbp+0x68]
       mov      ecx, dword ptr [rbp+0x70]
       mov      r8d, dword ptr [rbp+0x78]
       mov      r9d, dword ptr [rbp-0x74]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd512(ptr,ptr,int,int,int,int,int,int)]
       jmp      SHORT G_M000_IG90
 
G_M000_IG89:                ;; offset=0x0F56
       mov      rdi, 0x7A8492761830
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x7C]
       mov      dword ptr [rsp], eax
       mov      eax, dword ptr [rbp-0x80]
       mov      dword ptr [rsp+0x08], eax
       mov      rdi, qword ptr [rbp-0x98]
       mov      rsi, qword ptr [rbp-0xA0]
       mov      edx, dword ptr [rbp+0x68]
       mov      ecx, dword ptr [rbp+0x70]
       mov      r8d, dword ptr [rbp+0x78]
       mov      r9d, dword ptr [rbp-0x74]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
 
G_M000_IG90:                ;; offset=0x0F94
       mov      rdi, 0x7A8492761834
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
       mov      bword ptr [rbp-0xA8], rax
       xor      eax, eax
       mov      bword ptr [rbp-0xB0], rax
       xor      eax, eax
       mov      bword ptr [rbp-0xB8], rax
       xor      eax, eax
       mov      bword ptr [rbp-0xC0], rax
       mov      eax, dword ptr [rbp-0x7C]
       add      eax, 8
       mov      dword ptr [rbp-0x7C], eax
 
G_M000_IG91:                ;; offset=0x0FD0
       mov      eax, dword ptr [rbp-0x418]
       dec      eax
       mov      dword ptr [rbp-0x418], eax
       cmp      dword ptr [rbp-0x418], 0
       jg       SHORT G_M000_IG93
 
G_M000_IG92:                ;; offset=0x0FE7
       lea      rdi, [rbp-0x418]
       mov      esi, 943
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG93:                ;; offset=0x0FF8
       mov      eax, dword ptr [rbp-0x7C]
       cmp      eax, dword ptr [rbp-0x78]
       jl       G_M000_IG84
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3F0], rax
       mov      qword ptr [rbp-0x3E8], rdx
       mov      rdi, bword ptr [rbp-0x3F0]
       mov      rsi, qword ptr [rbp-0x3E8]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG95
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x400], rax
       mov      qword ptr [rbp-0x3F8], rdx
       mov      rdi, bword ptr [rbp-0x400]
       mov      rsi, qword ptr [rbp-0x3F8]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG99
       mov      rdi, bword ptr [rbp-0x58]
       mov      rsi, qword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG98
       mov      rdi, bword ptr [rbp+0x10]
       mov      rsi, qword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG100
 
G_M000_IG94:                ;; offset=0x1090
       mov      rdi, 0x7A8492761838
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG95:                ;; offset=0x109F
       mov      rdi, 0x7A849276183C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG96:                ;; offset=0x10AE
       xor      eax, eax
 
G_M000_IG97:                ;; offset=0x10B0
       add      rsp, 0x480
       pop      rbp
       ret      
 
G_M000_IG98:                ;; offset=0x10B9
       mov      rdi, 0x7A8492761840
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG95
 
G_M000_IG99:                ;; offset=0x10CA
       mov      rdi, 0x7A8492761844
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG95
 
G_M000_IG100:                ;; offset=0x10DB
       mov      rdi, 0x7A8492761848
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x410], rax
       mov      qword ptr [rbp-0x408], rdx
       lea      rdi, [rsp]
       lea      rsi, [rbp+0x10]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       mov      eax, dword ptr [rbp+0x68]
       mov      dword ptr [rsp+0x10], eax
       mov      eax, dword ptr [rbp+0x70]
       imul     eax, dword ptr [rbp+0x78]
       mov      dword ptr [rsp+0x18], eax
       mov      eax, dword ptr [rbp+0x80]
       mov      dword ptr [rsp+0x20], eax
       movzx    rax, byte  ptr [rbp+0x88]
       mov      dword ptr [rsp+0x28], eax
       mov      rdi, bword ptr [rbp-0x410]
       mov      rsi, qword ptr [rbp-0x408]
       mov      rdx, bword ptr [rbp+0x20]
       mov      rcx, qword ptr [rbp+0x28]
       mov      r8, bword ptr [rbp-0x58]
       mov      r9, qword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG101:                ;; offset=0x116F
       add      rsp, 0x480
       pop      rbp
       ret      
 
G_M000_IG102:                ;; offset=0x1178
       call     CORINFO_HELP_OVERFLOW
       int3     
 
; Total bytes of code 4478

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 220
; 1 inlinees with PGO data; 0 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 16
       lea      rbp, [rsp+0x10]
       xor      eax, eax
       mov      qword ptr [rbp-0x08], rax
 
G_M000_IG02:                ;; offset=0x0010
       xor      eax, eax
       xor      rcx, rcx
       test     esi, esi
       cmovne   rcx, rdi
       mov      bword ptr [rbp-0x08], rcx
       vbroadcastss zmm0, dword ptr [reloc @RWD00]
       vbroadcastss zmm1, dword ptr [reloc @RWD04]
       lea      edx, [rsi-0x10]
       test     edx, edx
       jl       SHORT G_M000_IG05
       align    [0 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0039
       movsxd   r8, eax
       vpandd   zmm2, zmm0, zmmword ptr [rcx+4*r8]
       vcmpgtps k1, zmm2, zmm1
       kmovw    r8d, k1
       test     r8, r8
       jne      SHORT G_M000_IG11
 
G_M000_IG04:                ;; offset=0x0053
       add      eax, 16
       cmp      eax, edx
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x005A
       xor      ecx, ecx
       mov      bword ptr [rbp-0x08], rcx
       cmp      eax, esi
       jl       SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0064
       mov      eax, 1
 
G_M000_IG07:                ;; offset=0x0069
       vzeroupper 
       add      rsp, 16
       pop      rbp
       ret      
 
G_M000_IG08:                ;; offset=0x0072
       test     eax, eax
       jl       SHORT G_M000_IG13
 
G_M000_IG09:                ;; offset=0x0076
       mov      ecx, eax
       vmovss   xmm0, dword ptr [rdi+4*rcx]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       ja       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x008F
       inc      eax
       cmp      eax, esi
       jl       SHORT G_M000_IG09
       jmp      SHORT G_M000_IG06
 
G_M000_IG11:                ;; offset=0x0097
       xor      eax, eax
 
G_M000_IG12:                ;; offset=0x0099
       vzeroupper 
       add      rsp, 16
       pop      rbp
       ret      
 
G_M000_IG13:                ;; offset=0x00A2
       cmp      eax, esi
       jae      SHORT G_M000_IG15
       mov      ecx, eax
       vmovss   xmm0, dword ptr [rdi+4*rcx]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       ja       SHORT G_M000_IG11
 
G_M000_IG14:                ;; offset=0x00BF
       inc      eax
       cmp      eax, esi
       jl       SHORT G_M000_IG13
       jmp      SHORT G_M000_IG06
 
G_M000_IG15:                ;; offset=0x00C7
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	7E7FFFFFh		; 8.50706e+37
RWD08  	dd	00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 205

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Dynamic PGO
; rbp based frame
; fully interruptible
; with Dynamic PGO: fgCalledCount is 72

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 16
       lea      rbp, [rsp+0x10]
       mov      rax, bword ptr [rbp+0x10]
 
G_M000_IG02:                ;; offset=0x000E
       xor      r10d, r10d
       mov      dword ptr [rax], r10d
 
G_M000_IG03:                ;; offset=0x0014
       mov      dword ptr [r9], r10d
 
G_M000_IG04:                ;; offset=0x0017
       mov      dword ptr [r8], r10d
       test     edi, edi
       jle      SHORT G_M000_IG10
       test     esi, esi
       jle      SHORT G_M000_IG10
       test     edx, edx
       jle      SHORT G_M000_IG10
       test     ecx, ecx
       jle      SHORT G_M000_IG10
 
G_M000_IG05:                ;; offset=0x002A
       mov      edi, edi
       imul     rdi, rdi, 16
       jo       SHORT G_M000_IG07
       imul     rdi, rdi, 8
       jo       SHORT G_M000_IG07
       mov      r10d, esi
       imul     r10, r10, 16
       jo       SHORT G_M000_IG07
       imul     r10, r10, 8
       jo       SHORT G_M000_IG07
       mov      esi, esi
       mov      edx, edx
       imul     rdx, rsi
       jo       SHORT G_M000_IG07
       mov      ecx, ecx
       imul     rcx, rdx
       jo       SHORT G_M000_IG07
       mov      rdx, rdi
       add      rdx, r10
       jo       SHORT G_M000_IG07
       add      rdx, rcx
       jo       SHORT G_M000_IG07
       cmp      rdx, 0x1000000
       jg       SHORT G_M000_IG06
       mov      dword ptr [r8], edi
       mov      dword ptr [r9], r10d
       mov      dword ptr [rax], ecx
       mov      dword ptr [rbp-0x04], 1
       jmp      SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0080
       xor      eax, eax
       mov      dword ptr [rbp-0x04], eax
       jmp      SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x0087
       call     CORINFO_HELP_OVERFLOW
       int3     
 
G_M000_IG08:                ;; offset=0x008D
       mov      eax, dword ptr [rbp-0x04]
 
G_M000_IG09:                ;; offset=0x0090
       add      rsp, 16
       pop      rbp
       ret      
 
G_M000_IG10:                ;; offset=0x0096
       xor      eax, eax
 
G_M000_IG11:                ;; offset=0x0098
       add      rsp, 16
       pop      rbp
       ret      
 
G_M000_IG12:                ;; offset=0x009E
       push     rax
 
G_M000_IG13:                ;; offset=0x009F
       xor      eax, eax
       mov      dword ptr [rbp-0x04], eax
       lea      rax, G_M000_IG08
 
G_M000_IG14:                ;; offset=0x00AB
       add      rsp, 8
       ret      
 
; Total bytes of code 176

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x3af
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 101
; 44 inlinees with PGO data; 165 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 224
       mov      qword ptr [rsp+0x568], r15
       mov      qword ptr [rsp+0x560], r14
       mov      qword ptr [rsp+0x558], r13
       mov      qword ptr [rsp+0x550], r12
       mov      qword ptr [rsp+0x548], rbx
       lea      rbp, [rsp+0xE0]
       xor      eax, eax
       mov      qword ptr [rbp-0x30], rax
       mov      qword ptr [rbp-0x38], rax
       mov      r12d, dword ptr [rbp+0x4F0]
       mov      r15d, dword ptr [rbp+0x4F8]
       mov      r14d, dword ptr [rbp+0x500]
       mov      ebx, dword ptr [rbp+0x508]
       mov      r13d, dword ptr [rbp+0x510]
       mov      r10d, dword ptr [rbp+0x41C]
       mov      r9d, dword ptr [rbp+0x418]
       mov      r11d, dword ptr [rbp+0x414]
 
G_M000_IG02:                ;; offset=0x007D
       mov      r8, bword ptr [rbp+0x448]
       mov      bword ptr [rbp-0x70], r8
       mov      ecx, dword ptr [rbp+0x450]
       mov      dword ptr [rbp-0x44], ecx
       mov      rdx, bword ptr [rbp+0x438]
       mov      bword ptr [rbp-0x78], rdx
       mov      esi, dword ptr [rbp+0x440]
       mov      dword ptr [rbp-0x48], esi
       mov      rdi, bword ptr [rbp+0x4C0]
       mov      bword ptr [rbp-0x80], rdi
       mov      eax, dword ptr [rbp+0x4C8]
       mov      dword ptr [rbp-0x4C], eax
       mov      rdx, bword ptr [rbp+0x4D0]
       mov      bword ptr [rbp-0x88], rdx
       mov      esi, dword ptr [rbp+0x4D8]
       mov      dword ptr [rbp-0x50], esi
       mov      r8, bword ptr [rbp+0x4E0]
       mov      bword ptr [rbp-0x90], r8
       mov      ecx, dword ptr [rbp+0x4E8]
       mov      dword ptr [rbp-0x54], ecx
       mov      r8, bword ptr [rbp+0x458]
       mov      bword ptr [rbp-0x98], r8
       mov      r8d, dword ptr [rbp+0x460]
       mov      dword ptr [rbp-0x58], r8d
       mov      r8, bword ptr [rbp+0x4A0]
       mov      bword ptr [rbp-0xA0], r8
       mov      r8d, dword ptr [rbp+0x4A8]
       mov      dword ptr [rbp-0x5C], r8d
       mov      r8, bword ptr [rbp+0x4B0]
       mov      bword ptr [rbp-0xA8], r8
       mov      r8d, dword ptr [rbp+0x4B8]
       mov      dword ptr [rbp-0x60], r8d
       mov      r8d, r14d
       imul     r8d, ebx
       mov      dword ptr [rbp-0x64], r8d
       cmp      r11d, r9d
       jl       SHORT G_M000_IG09
       jmp      G_M000_IG39
 
G_M000_IG03:                ;; offset=0x0147
       mov      dword ptr [rbp+0x4F0], r12d
       mov      r11d, dword ptr [rbp+0x414]
 
G_M000_IG04:                ;; offset=0x0155
       xor      edi, edi
       mov      bword ptr [rbp+0x3E8], rdi
 
G_M000_IG05:                ;; offset=0x015E
       mov      bword ptr [rbp+0x3E0], rdi
 
G_M000_IG06:                ;; offset=0x0165
       mov      bword ptr [rbp+0x3D8], rdi
 
G_M000_IG07:                ;; offset=0x016C
       mov      bword ptr [rbp+0x3D0], rdi
       add      r11d, 8
       mov      edi, dword ptr [rbp+0x418]
       cmp      r11d, edi
       mov      r9d, edi
       mov      eax, dword ptr [rbp-0x4C]
       mov      r10d, dword ptr [rbp+0x41C]
       mov      r12d, dword ptr [rbp+0x4F0]
       jge      G_M000_IG37
 
G_M000_IG08:                ;; offset=0x019A
       mov      r13d, dword ptr [rbp+0x510]
 
G_M000_IG09:                ;; offset=0x01A1
       mov      dword ptr [rbp+0x418], r9d
       mov      r8d, r9d
       sub      r8d, r11d
       cmp      r8d, 8
       jl       G_M000_IG40
       mov      r8d, 8
 
G_M000_IG10:                ;; offset=0x01BE
       mov      dword ptr [rbp+0x410], r8d
       mov      dword ptr [rsp], ebx
       mov      dword ptr [rbp+0x41C], r10d
       mov      dword ptr [rsp+0x08], r10d
       mov      dword ptr [rbp+0x414], r11d
       mov      dword ptr [rsp+0x10], r11d
       mov      dword ptr [rsp+0x18], r8d
       mov      rdi, bword ptr [rbp-0x98]
       mov      esi, dword ptr [rbp-0x58]
       mov      rdx, bword ptr [rbp-0x80]
       mov      ecx, eax
       mov      r8d, r12d
       mov      r9d, r14d
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      eax, dword ptr [rbp-0x4C]
       mov      edi, eax
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG11:                ;; offset=0x0215
       test     edi, edi
       je       SHORT G_M000_IG12
       mov      r9, bword ptr [rbp-0x80]
       mov      rsi, r9
 
G_M000_IG12:                ;; offset=0x0220
       mov      bword ptr [rbp-0x30], rsi
       xor      edi, edi
       cmp      eax, 8
       jl       SHORT G_M000_IG14
       align    [0 bytes for IG13]
 
G_M000_IG13:                ;; offset=0x022B
       mov      edx, edi
       sar      edx, 31
       and      edx, 7
       add      edx, edi
       sar      edx, 3
       movsxd   rdx, edx
       shl      rdx, 5
       vpand    ymm1, ymm0, ymmword ptr [rdx+rsi]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG41
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, eax
       jle      SHORT G_M000_IG13
       align    [0 bytes for IG14]
 
G_M000_IG14:                ;; offset=0x025D
       cmp      edi, eax
       jl       G_M000_IG42
       xor      edi, edi
       mov      bword ptr [rbp-0x30], rdi
       mov      edi, 1
 
G_M000_IG15:                ;; offset=0x0270
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       test     edi, edi
       je       G_M000_IG48
       xor      rdi, rdi
       test     eax, eax
       je       SHORT G_M000_IG16
       mov      r9, bword ptr [rbp-0x80]
       mov      rdi, r9
 
G_M000_IG16:                ;; offset=0x028B
       mov      bword ptr [rbp+0x3E8], rdi
       xor      rsi, rsi
       cmp      dword ptr [rbp-0x44], 0
       je       SHORT G_M000_IG17
       mov      r11, bword ptr [rbp-0x70]
       mov      rsi, r11
 
G_M000_IG17:                ;; offset=0x02A1
       mov      bword ptr [rbp+0x3E0], rsi
       xor      r8, r8
       cmp      dword ptr [rbp-0x50], 0
       je       SHORT G_M000_IG18
       mov      rdx, bword ptr [rbp-0x88]
       mov      r8, rdx
 
G_M000_IG18:                ;; offset=0x02BB
       mov      bword ptr [rbp+0x3D8], r8
       mov      qword ptr [rbp+0x3F8], r8
       xor      r10, r10
       cmp      dword ptr [rbp-0x54], 0
       je       SHORT G_M000_IG19
       mov      r10, bword ptr [rbp-0x90]
       mov      r11, r10
       mov      r10, r11
 
G_M000_IG19:                ;; offset=0x02DF
       mov      bword ptr [rbp+0x3D0], r10
       mov      qword ptr [rbp+0x3F0], r10
       cmp      r13d, 16
       jne      G_M000_IG43
       mov      rdx, r8
       mov      ecx, r12d
       mov      r8d, r15d
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int)]
 
G_M000_IG20:                ;; offset=0x0306
       mov      r10d, dword ptr [rbp-0x50]
       mov      eax, r10d
       xor      rdx, rdx
       mov      bword ptr [rbp-0x38], rdx
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG21:                ;; offset=0x031C
       test     eax, eax
       je       SHORT G_M000_IG22
       mov      r11, bword ptr [rbp-0x88]
       mov      rdx, r11
 
G_M000_IG22:                ;; offset=0x032A
       mov      bword ptr [rbp-0x38], rdx
       xor      eax, eax
       cmp      r10d, 8
       jl       SHORT G_M000_IG24
       align    [0 bytes for IG23]
 
G_M000_IG23:                ;; offset=0x0336
       mov      edi, eax
       sar      edi, 31
       and      edi, 7
       add      edi, eax
       sar      edi, 3
       movsxd   rdi, edi
       shl      rdi, 5
       vpand    ymm1, ymm0, ymmword ptr [rdi+rdx]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG44
       add      eax, 8
       lea      edi, [rax+0x08]
       cmp      edi, r10d
       jle      SHORT G_M000_IG23
       align    [0 bytes for IG24]
 
G_M000_IG24:                ;; offset=0x0369
       cmp      eax, r10d
       jl       G_M000_IG45
       xor      eax, eax
       mov      bword ptr [rbp-0x38], rax
       mov      eax, 1
 
G_M000_IG25:                ;; offset=0x037D
       xor      rdx, rdx
       mov      bword ptr [rbp-0x38], rdx
       test     eax, eax
       je       G_M000_IG48
       mov      dword ptr [rbp+0x510], r13d
       cmp      r13d, 16
       jne      G_M000_IG47
       mov      r9d, dword ptr [rbp-0x64]
       mov      edi, r9d
       xor      esi, esi
       cmp      esi, r15d
       jl       SHORT G_M000_IG29
       jmp      G_M000_IG03
 
G_M000_IG26:                ;; offset=0x03AF
       mov      r11d, dword ptr [rbp+0x414]
       mov      r12d, dword ptr [rbp+0x410]
       mov      r13, qword ptr [rbp+0x3F0]
 
G_M000_IG27:                ;; offset=0x03C4
       add      esi, 16
       cmp      esi, r15d
       jge      G_M000_IG04
 
G_M000_IG28:                ;; offset=0x03D0
       mov      dword ptr [rbp+0x414], r11d
       mov      r12d, dword ptr [rbp+0x4F0]
 
G_M000_IG29:                ;; offset=0x03DE
       xor      ecx, ecx
       cmp      ecx, dword ptr [rbp+0x410]
       mov      dword ptr [rbp+0x4F0], r12d
       jge      SHORT G_M000_IG26
       align    [0 bytes for IG30]
 
G_M000_IG30:                ;; offset=0x03EF
       lea      eax, [8*rsi]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp+0x3F8]
       mov      r9d, ecx
       shl      r9d, 4
       movsxd   r9, r9d
       lea      r9, [rax+4*r9]
       lea      eax, [8*r15]
       mov      dword ptr [rbp-0x3C], eax
       mov      r11d, dword ptr [rbp+0x414]
       lea      r12d, [r11+rcx]
       mov      r13d, dword ptr [rbp+0x41C]
       mov      eax, r12d
       cdq      
       idiv     edx:eax, r13d
       lea      edx, [rax+rax]
       mov      dword ptr [rbp-0x40], edx
       mov      dword ptr [rbp+0x41C], r13d
       mov      eax, r12d
       cdq      
       idiv     edx:eax, r13d
       add      edx, edx
       vmovups  zmm0, zmmword ptr [r9]
       mov      r12d, dword ptr [rbp-0x3C]
       lea      eax, [4*r12]
       cdqe     
       vmovups  zmm1, zmmword ptr [r9+4*rax]
       vaddps   zmm0, zmm0, zmm1
       lea      eax, [8*r12]
       cdqe     
       vmovups  zmm2, zmmword ptr [r9+4*rax]
       vaddps   zmm0, zmm0, zmm2
       vsubps   zmm1, zmm1, zmm2
       lea      eax, [r12+2*r12]
       lea      r8d, [4*rax]
       movsxd   r8, r8d
       vsubps   zmm1, zmm1, zmmword ptr [r9+4*r8]
       movsxd   r8, r12d
       vmovups  zmm2, zmmword ptr [r9+4*r8]
       lea      r8d, [r12+4*r12]
       movsxd   r13, r8d
       vmovups  zmm3, zmmword ptr [r9+4*r13]
       vaddps   zmm2, zmm2, zmm3
       lea      r13d, [r12+8*r12]
       movsxd   r13, r13d
       vmovups  zmm4, zmmword ptr [r9+4*r13]
       vaddps   zmm2, zmm2, zmm4
       vsubps   zmm3, zmm3, zmm4
       imul     r13d, r12d, 13
       movsxd   r13, r13d
       vsubps   zmm3, zmm3, zmmword ptr [r9+4*r13]
       lea      r13d, [r12+r12]
       movsxd   r13, r13d
       vmovups  zmm4, zmmword ptr [r9+4*r13]
       lea      r13d, [rax+rax]
       movsxd   r13, r13d
       vmovups  zmm5, zmmword ptr [r9+4*r13]
       vaddps   zmm4, zmm4, zmm5
       add      r8d, r8d
       movsxd   r8, r8d
       vmovups  zmm6, zmmword ptr [r9+4*r8]
       vaddps   zmm4, zmm6, zmm4
       vsubps   zmm5, zmm5, zmm6
       imul     r8d, r12d, 14
       movsxd   r8, r8d
 
G_M000_IG31:                ;; offset=0x0527
       vsubps   zmm5, zmm5, zmmword ptr [r9+4*r8]
       movsxd   r8, eax
       vmovups  zmm6, zmmword ptr [r9+4*r8]
       lea      r8d, [8*r12]
       sub      r8d, r12d
       movsxd   r8, r8d
       vmovups  zmm7, zmmword ptr [r9+4*r8]
       vaddps   zmm6, zmm6, zmm7
       imul     r8d, r12d, 11
       movsxd   r8, r8d
       vmovups  zmm8, zmmword ptr [r9+4*r8]
       vaddps   zmm6, zmm6, zmm8
       vsubps   zmm7, zmm7, zmm8
       mov      r8d, r12d
       shl      r8d, 4
       sub      r8d, r12d
       movsxd   r8, r8d
       vsubps   zmm7, zmm7, zmmword ptr [r9+4*r8]
       vaddps   zmm0, zmm2, zmm0
       vaddps   zmm0, zmm0, zmm4
       vsubps   zmm2, zmm2, zmm4
       vsubps   zmm2, zmm2, zmm6
       vaddps   zmm1, zmm1, zmm3
       vaddps   zmm1, zmm1, zmm5
       vsubps   zmm3, zmm3, zmm5
       vsubps   zmm3, zmm3, zmm7
       mov      r8d, dword ptr [rbp-0x40]
       mov      r9d, r8d
       imul     r9d, ebx
       add      r9d, edx
       shl      r9d, 4
       movsxd   r9, r9d
       mov      eax, esi
       imul     eax, edi
       cdqe     
       shl      rax, 2
       mov      r13, qword ptr [rbp+0x3F0]
       add      rax, r13
       lea      r9, [rax+4*r9]
       vmovups  zmmword ptr [r9], zmm0
       lea      eax, [rdx+0x01]
       cmp      eax, ebx
       jge      SHORT G_M000_IG33
 
G_M000_IG32:                ;; offset=0x05EC
       vmovups  zmmword ptr [r9+0x40], zmm2
 
G_M000_IG33:                ;; offset=0x05F3
       inc      r8d
       cmp      r8d, r14d
       jge      SHORT G_M000_IG35
 
G_M000_IG34:                ;; offset=0x05FB
       mov      r8d, ebx
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [r9+4*r8], zmm1
       inc      edx
       cmp      edx, ebx
       jge      SHORT G_M000_IG35
       lea      edx, [rbx+0x01]
       shl      edx, 4
       movsxd   rdx, edx
       vmovups  zmmword ptr [r9+4*rdx], zmm3
 
G_M000_IG35:                ;; offset=0x0622
       inc      ecx
       mov      r12d, dword ptr [rbp+0x410]
       cmp      ecx, r12d
       jge      G_M000_IG27
 
G_M000_IG36:                ;; offset=0x0634
       mov      dword ptr [rbp+0x414], r11d
       jmp      G_M000_IG30
 
G_M000_IG37:                ;; offset=0x0640
       mov      rdi, bword ptr [rbp-0x90]
       mov      esi, dword ptr [rbp-0x54]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG48
       mov      rdi, bword ptr [rbp-0x90]
       mov      esi, dword ptr [rbp-0x54]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG48
       mov      rdi, bword ptr [rbp-0x78]
       mov      esi, dword ptr [rbp-0x48]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG48
       mov      rdi, bword ptr [rbp-0xA0]
       mov      esi, dword ptr [rbp-0x5C]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG48
       lea      rdi, [rsp]
       lea      rsi, [rbp+0x4A0]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       mov      dword ptr [rsp+0x10], r15d
       mov      ebx, dword ptr [rbp-0x64]
       mov      dword ptr [rsp+0x18], ebx
       mov      r13d, dword ptr [rbp+0x510]
       mov      dword ptr [rsp+0x20], r13d
       movzx    r8, byte  ptr [rbp+0x518]
       mov      dword ptr [rsp+0x28], r8d
       mov      r8, bword ptr [rbp-0x78]
       mov      r9d, dword ptr [rbp-0x48]
       mov      rdx, bword ptr [rbp-0xA8]
       mov      ecx, dword ptr [rbp-0x60]
       mov      rdi, bword ptr [rbp-0x90]
       mov      esi, dword ptr [rbp-0x54]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG38:                ;; offset=0x0705
       vzeroupper 
       add      rsp, 0x548
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG39:                ;; offset=0x071A
       mov      dword ptr [rbp+0x510], r13d
       jmp      G_M000_IG37
 
G_M000_IG40:                ;; offset=0x0726
       jmp      G_M000_IG10
 
G_M000_IG41:                ;; offset=0x072B
       xor      edi, edi
       jmp      G_M000_IG15
 
G_M000_IG42:                ;; offset=0x0732
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      ecx, 0x7F800000
       andn     edx, edx, ecx
       je       SHORT G_M000_IG41
       inc      edi
       jmp      G_M000_IG14
 
G_M000_IG43:                ;; offset=0x074B
       mov      rdx, r8
       mov      dword ptr [rbp+0x4F0], r12d
       mov      ecx, r12d
       mov      r8d, r15d
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int)]
       mov      r12d, dword ptr [rbp+0x4F0]
       jmp      G_M000_IG20
 
G_M000_IG44:                ;; offset=0x076D
       xor      eax, eax
       jmp      G_M000_IG25
 
G_M000_IG45:                ;; offset=0x0774
       movsxd   rdi, eax
       mov      edi, dword ptr [rdx+4*rdi]
       mov      esi, 0x7F800000
       andn     edi, edi, esi
       je       SHORT G_M000_IG46
       inc      eax
       mov      r10d, dword ptr [rbp-0x50]
       jmp      G_M000_IG24
 
G_M000_IG46:                ;; offset=0x0791
       mov      r10d, dword ptr [rbp-0x50]
       jmp      SHORT G_M000_IG44
 
G_M000_IG47:                ;; offset=0x0797
       mov      eax, dword ptr [rbp+0x414]
       mov      dword ptr [rsp], eax
       mov      edi, dword ptr [rbp+0x410]
       mov      dword ptr [rsp+0x08], edi
       mov      rdi, qword ptr [rbp+0x3F8]
       mov      rsi, qword ptr [rbp+0x3F0]
       mov      edx, r15d
       mov      ecx, r14d
       mov      r8d, ebx
       mov      r9d, dword ptr [rbp+0x41C]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
       mov      dword ptr [rbp+0x4F0], r12d
       mov      r11d, dword ptr [rbp+0x414]
       jmp      G_M000_IG04
 
G_M000_IG48:                ;; offset=0x07E1
       xor      eax, eax
 
G_M000_IG49:                ;; offset=0x07E3
       vzeroupper 
       add      rsp, 0x548
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
RWD00  	dd	7F800000h		;       inf

; Total bytes of code 2040

; Assembly listing for method KernelAccess:PrepareWinograd(System.ReadOnlySpan`1[float],int,int,int):float[] (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; partially interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 48
       lea      rbp, [rsp+0x30]
       xor      eax, eax
       mov      qword ptr [rbp-0x28], rax
       mov      qword ptr [rbp-0x30], rax
       mov      bword ptr [rbp-0x10], rdi
       mov      qword ptr [rbp-0x08], rsi
       mov      dword ptr [rbp-0x14], edx
       mov      dword ptr [rbp-0x18], ecx
       mov      dword ptr [rbp-0x1C], r8d
 
G_M000_IG02:                ;; offset=0x0026
       mov      rax, 0x7A7CAD800190
       mov      rax, gword ptr [rax]
       mov      gword ptr [rbp-0x28], rax
       mov      rdi, gword ptr [rbp-0x28]
       mov      rsi, 0x7A8492919930
       call     CORINFO_HELP_DELEGATEPROFILE32
       mov      rax, gword ptr [rbp-0x28]
       mov      gword ptr [rbp-0x30], rax
       mov      rax, gword ptr [rbp-0x30]
       mov      rsi, bword ptr [rbp-0x10]
       mov      rdx, qword ptr [rbp-0x08]
       mov      ecx, dword ptr [rbp-0x14]
       mov      r8d, dword ptr [rbp-0x18]
       mov      r9d, dword ptr [rbp-0x1C]
       mov      rdi, gword ptr [rax+0x08]
       mov      rax, gword ptr [rbp-0x30]
       call     [rax+0x18]KernelAccess+PrepareCall:Invoke(System.ReadOnlySpan`1[float],int,int,int):float[]:this
       nop      
 
G_M000_IG03:                ;; offset=0x0075
       add      rsp, 48
       pop      rbp
       ret      
 
; Total bytes of code 123

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
       mov      rdi, 0x7A84927580D0
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
       mov      rdi, 0x7A84927580D4
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
       mov      rdi, 0x7A84927580D8
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
       mov      rdi, 0x7A84927580DC
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
       mov      rdi, 0x7A84927580E0
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
       mov      rdi, 0x7A84927580E4
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
       mov      rdi, 0x7A84927580E8
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
       mov      rdi, 0x7A84927580EC
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
       mov      rdi, 0x7A84927580F0
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
       mov      rdi, 0x7A84927580F4
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
       mov      rdi, 0x7A84927580F8
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
       mov      rdi, 0x7A84927580FC
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
       mov      rdi, 0x7A8492758100
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
       mov      rdi, 0x7A8492758104
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
       mov      rdi, 0x7A8492758108
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
       mov      rdi, 0x7A849275810C
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
       mov      rdi, 0x7A8492758110
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
       mov      rdi, 0x7A8492758114
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
       mov      rdi, 0x7A8492758118
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
       mov      rdi, 0x7A849275811C
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG54:                ;; offset=0x0EA0
       vzeroupper 
       add      rsp, 0x780
       pop      rbp
       ret      
 
; Total bytes of code 3756

; Assembly listing for method KernelAccess:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; rsp based frame
; partially interruptible

G_M000_IG01:                ;; offset=0x0000
       push     rbx
       sub      rsp, 16
       mov      r10d, edx
       mov      r11d, ecx
       mov      rax, r8
       mov      r8d, esi
 
G_M000_IG02:                ;; offset=0x0011
       mov      bword ptr [rsp], r9
       mov      rsi, bword ptr [rsp+0x20]
       mov      bword ptr [rsp+0x08], rsi
       mov      rsi, 0x7A7CAD800198
       mov      rbx, gword ptr [rsi]
       mov      esi, edi
       mov      edx, r8d
       mov      ecx, r10d
       mov      r8d, r11d
       mov      r9, rax
       mov      rdi, gword ptr [rbx+0x08]
       call     [rbx+0x18]KernelAccess+PlanCall:Invoke(int,int,int,int,byref,byref,byref):bool:this
       nop      
 
G_M000_IG03:                ;; offset=0x0042
       add      rsp, 16
       pop      rbx
       ret      
 
; Total bytes of code 72

; Assembly listing for method KernelAccess:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; rbp based frame
; partially interruptible

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 144
       lea      rbp, [rsp+0x90]
       mov      rax, rdx
       mov      edx, esi
       mov      r10d, ecx
 
G_M000_IG02:                ;; offset=0x0018
       mov      bword ptr [rsp], r8
       mov      dword ptr [rsp+0x08], r9d
       mov      rsi, bword ptr [rbp+0x10]
       mov      bword ptr [rsp+0x10], rsi
       mov      ecx, dword ptr [rbp+0x18]
       mov      dword ptr [rsp+0x18], ecx
       mov      rsi, bword ptr [rbp+0x20]
       mov      bword ptr [rsp+0x20], rsi
       mov      ecx, dword ptr [rbp+0x28]
       mov      dword ptr [rsp+0x28], ecx
       mov      rsi, bword ptr [rbp+0x30]
       mov      bword ptr [rsp+0x30], rsi
       mov      ecx, dword ptr [rbp+0x38]
       mov      dword ptr [rsp+0x38], ecx
       mov      rsi, bword ptr [rbp+0x40]
       mov      bword ptr [rsp+0x40], rsi
       mov      ecx, dword ptr [rbp+0x48]
       mov      dword ptr [rsp+0x48], ecx
       mov      rsi, bword ptr [rbp+0x50]
       mov      bword ptr [rsp+0x50], rsi
       mov      ecx, dword ptr [rbp+0x58]
       mov      dword ptr [rsp+0x58], ecx
       mov      esi, dword ptr [rbp+0x68]
       mov      dword ptr [rsp+0x60], esi
       mov      esi, dword ptr [rbp+0x70]
       mov      dword ptr [rsp+0x68], esi
       mov      esi, dword ptr [rbp+0x78]
       mov      dword ptr [rsp+0x70], esi
       mov      esi, dword ptr [rbp+0x80]
       mov      dword ptr [rsp+0x78], esi
       movzx    rsi, byte  ptr [rbp+0x88]
       mov      dword ptr [rsp+0x80], esi
       mov      rsi, rdi
       mov      rcx, rax
       mov      r8d, r10d
       mov      r9, 0x7A7CAD8001A8
       mov      rax, gword ptr [r9]
       mov      r9d, dword ptr [rbp+0x60]
       mov      rdi, gword ptr [rax+0x08]
       call     [rax+0x18]KernelAccess+WinogradCall:Invoke(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool:this
       nop      
 
G_M000_IG03:                ;; offset=0x00C1
       add      rsp, 144
       pop      rbp
       ret      
 
; Total bytes of code 202

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100
; 44 inlinees with PGO data; 165 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 456
       lea      rbp, [rsp+0x1F0]
       vxorps   xmm8, xmm8, xmm8
       vmovdqu32 zmmword ptr [rbp-0x170], zmm8
       vmovdqu32 zmmword ptr [rbp-0x130], zmm8
       vmovdqu32 zmmword ptr [rbp-0xF0], zmm8
       vmovdqu32 zmmword ptr [rbp-0xB0], zmm8
       vmovdqu32 zmmword ptr [rbp-0x70], zmm8
       xor      eax, eax
       mov      qword ptr [rbp-0x30], rax
       mov      bword ptr [rbp-0x1A8], rdi
       mov      dword ptr [rbp-0x17C], esi
       mov      bword ptr [rbp-0x1B0], rdx
       mov      dword ptr [rbp-0x180], ecx
       mov      bword ptr [rbp-0x1B8], r8
       mov      dword ptr [rbp-0x184], r9d
       mov      eax, dword ptr [rbp+0x60]
       mov      r13d, dword ptr [rbp+0x68]
       mov      r12d, dword ptr [rbp+0x70]
       mov      r14d, dword ptr [rbp+0x78]
       mov      ebx, dword ptr [rbp+0x38]
       mov      r15d, dword ptr [rbp+0x48]
 
G_M000_IG02:                ;; offset=0x0094
       mov      r8d, dword ptr [rbp+0x80]
       cmp      r8d, 8
       je       SHORT G_M000_IG03
       cmp      r8d, 16
       jne      G_M000_IG81
 
G_M000_IG03:                ;; offset=0x00AB
       cmp      eax, 16
       jl       G_M000_IG81
       test     al, 15
       jne      G_M000_IG81
       cmp      r13d, 32
       jl       G_M000_IG81
       test     r13b, 15
       jne      G_M000_IG81
       test     r12d, r12d
       jle      G_M000_IG81
       test     r14d, r14d
       setle    r9b
       movzx    r9, r9b
 
G_M000_IG04:                ;; offset=0x00E4
       test     r9d, r9d
       jne      G_M000_IG82
       mov      r9d, r12d
       add      r9d, 2
       jo       G_M000_IG113
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, r14d
       add      r9d, 2
       jo       G_M000_IG113
       imul     r9d, dword ptr [rbp-0x1BC]
       jo       G_M000_IG113
       mov      dword ptr [rbp+0x60], eax
       imul     r9d, eax
       jo       G_M000_IG113
       mov      r9d, r12d
       add      r9d, 1
       jo       G_M000_IG113
       mov      dword ptr [rbp-0x198], r9d
       sub      r9d, 1
       jo       G_M000_IG113
       imul     r9d, r13d
       jo       G_M000_IG113
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, r14d
       add      r9d, 1
       jo       G_M000_IG113
       sub      r9d, 1
       jo       G_M000_IG113
       imul     r9d, dword ptr [rbp-0x1BC]
       jo       G_M000_IG113
       mov      dword ptr [rbp+0x80], r8d
       imul     r9d, r8d, 2
       jo       G_M000_IG113
       mov      dword ptr [rbp-0x190], r9d
       add      r9d, r13d
       jo       G_M000_IG113
       mov      eax, r9d
       sub      eax, 1
       jo       G_M000_IG113
       mov      r9d, dword ptr [rbp-0x190]
       cdq      
       idiv     edx:eax, r9d
       imul     r9d, eax
       jo       G_M000_IG113
       mov      eax, dword ptr [rbp+0x60]
 
G_M000_IG05:                ;; offset=0x01C2
       imul     r9d, eax
       jo       G_M000_IG113
       imul     r9d, r9d, 9
       jo       G_M000_IG113
       lea      r9, [rbp-0x40]
       mov      qword ptr [rsp], r9
       lea      r8, [rbp-0x30]
       lea      r9, [rbp-0x38]
       mov      dword ptr [rbp+0x60], eax
       mov      edi, eax
       mov      esi, r13d
       mov      edx, r12d
       mov      ecx, r14d
       call     [Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool]
       test     eax, eax
       je       G_M000_IG111
       mov      eax, dword ptr [rbp+0x60]
       mov      edi, eax
       imul     edi, r12d
       jo       G_M000_IG113
       imul     edi, r14d
       jo       G_M000_IG113
       mov      ecx, dword ptr [rbp-0x17C]
       cmp      edi, ecx
       jne      G_M000_IG83
       mov      dword ptr [rbp+0x60], eax
       imul     edi, eax, 16
       jo       G_M000_IG113
       imul     edi, r13d
       jo       G_M000_IG113
       mov      edx, dword ptr [rbp-0x180]
       cmp      edi, edx
       jne      G_M000_IG83
       mov      esi, dword ptr [rbp+0x28]
       cmp      esi, dword ptr [rbp-0x40]
       jne      G_M000_IG83
       mov      r8d, dword ptr [rbp-0x184]
       test     r8d, r8d
       je       SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x0265
       mov      dword ptr [rbp-0x184], r8d
       cmp      r8d, r13d
       mov      r8d, dword ptr [rbp-0x184]
       jne      G_M000_IG83
 
G_M000_IG07:                ;; offset=0x027C
       mov      r9d, dword ptr [rbp+0x18]
       test     r9d, r9d
       je       SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x0285
       cmp      r9d, dword ptr [rbp-0x40]
       jne      G_M000_IG83
 
G_M000_IG09:                ;; offset=0x028F
       cmp      ebx, dword ptr [rbp-0x30]
       jl       G_M000_IG83
       cmp      r15d, dword ptr [rbp-0x38]
       jl       G_M000_IG83
       mov      r10d, dword ptr [rbp+0x58]
       cmp      r10d, dword ptr [rbp-0x40]
       jl       G_M000_IG83
       mov      edi, dword ptr [rbp-0x30]
       cmp      edi, ebx
       ja       G_M000_IG84
       mov      rbx, bword ptr [rbp+0x30]
       mov      r11d, edi
       mov      edi, dword ptr [rbp-0x38]
       cmp      edi, r15d
       ja       G_M000_IG84
       mov      r15, bword ptr [rbp+0x40]
       mov      esi, dword ptr [rbp-0x40]
       cmp      esi, r10d
       ja       G_M000_IG84
       mov      r10, bword ptr [rbp+0x50]
       test     ecx, ecx
       je       SHORT G_M000_IG10
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG10
       mov      dword ptr [rbp-0x184], r8d
       mov      r8, bword ptr [rbp+0x20]
       sub      r8, qword ptr [rbp-0x1A8]
       mov      qword ptr [rbp-0x88], r8
       mov      r8d, ecx
       shl      r8, 2
       cmp      qword ptr [rbp-0x88], r8
       jb       G_M000_IG88
       mov      r8d, dword ptr [rbp+0x28]
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0x88]
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG88
 
G_M000_IG10:                ;; offset=0x0338
       test     edx, edx
       je       SHORT G_M000_IG11
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG11
       mov      dword ptr [rbp-0x184], r8d
       mov      r8, bword ptr [rbp+0x20]
       sub      r8, qword ptr [rbp-0x1B0]
       mov      qword ptr [rbp-0x90], r8
       mov      r8d, edx
       shl      r8, 2
       cmp      qword ptr [rbp-0x90], r8
       jb       G_M000_IG88
       mov      r8d, dword ptr [rbp+0x28]
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0x90]
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG88
 
G_M000_IG11:                ;; offset=0x038E
       mov      dword ptr [rbp-0x184], r8d
       test     r8d, r8d
       je       SHORT G_M000_IG12
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG12
       mov      r8, bword ptr [rbp+0x20]
       sub      r8, qword ptr [rbp-0x1B8]
       mov      qword ptr [rbp-0x98], r8
       mov      r8d, dword ptr [rbp-0x184]
       shl      r8, 2
       cmp      qword ptr [rbp-0x98], r8
       jb       G_M000_IG88
       mov      r8d, dword ptr [rbp+0x28]
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0x98]
       jb       G_M000_IG88
 
G_M000_IG12:                ;; offset=0x03E2
       test     r9d, r9d
       je       SHORT G_M000_IG13
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG13
       mov      r8, bword ptr [rbp+0x20]
       sub      r8, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xA0], r8
       mov      dword ptr [rbp+0x18], r9d
       mov      r8d, r9d
       shl      r8, 2
       cmp      qword ptr [rbp-0xA0], r8
       jb       G_M000_IG88
       mov      r8d, dword ptr [rbp+0x28]
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xA0]
       mov      r9d, dword ptr [rbp+0x18]
       jb       G_M000_IG88
 
G_M000_IG13:                ;; offset=0x0430
       test     ecx, ecx
       je       SHORT G_M000_IG14
       test     r11d, r11d
       je       SHORT G_M000_IG14
       mov      dword ptr [rbp+0x18], r9d
       mov      r8, bword ptr [rbp-0x1A8]
       mov      r9, rbx
       sub      r9, r8
       mov      qword ptr [rbp-0xA8], r9
       mov      r9d, ecx
       shl      r9, 2
       cmp      qword ptr [rbp-0xA8], r9
       jb       G_M000_IG88
       mov      r9d, r11d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xA8]
       mov      bword ptr [rbp-0x1A8], r8
       mov      r9d, dword ptr [rbp+0x18]
       jb       G_M000_IG88
 
G_M000_IG14:                ;; offset=0x0487
       test     edx, edx
       je       SHORT G_M000_IG15
       test     r11d, r11d
       je       SHORT G_M000_IG15
       mov      dword ptr [rbp+0x18], r9d
       mov      r9, bword ptr [rbp-0x1B0]
       mov      r8, rbx
       sub      r8, r9
       mov      qword ptr [rbp-0xB0], r8
       mov      r8d, edx
       shl      r8, 2
       cmp      qword ptr [rbp-0xB0], r8
       jb       G_M000_IG88
       mov      r8d, r11d
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xB0]
       mov      bword ptr [rbp-0x1B0], r9
       jb       G_M000_IG88
       mov      r9d, dword ptr [rbp+0x18]
 
G_M000_IG15:                ;; offset=0x04DE
       mov      r8d, dword ptr [rbp-0x184]
       test     r8d, r8d
       je       SHORT G_M000_IG16
       test     r11d, r11d
       je       SHORT G_M000_IG16
       mov      dword ptr [rbp+0x18], r9d
       mov      r9, rbx
       sub      r9, qword ptr [rbp-0x1B8]
       mov      qword ptr [rbp-0xB8], r9
       mov      dword ptr [rbp-0x184], r8d
       mov      r9d, r8d
       shl      r9, 2
       cmp      qword ptr [rbp-0xB8], r9
       jb       G_M000_IG88
       mov      r9d, r11d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xB8]
       mov      r8d, dword ptr [rbp-0x184]
       mov      r9d, dword ptr [rbp+0x18]
       jb       G_M000_IG88
 
G_M000_IG16:                ;; offset=0x0541
       mov      dword ptr [rbp+0x18], r9d
       test     r9d, r9d
       je       SHORT G_M000_IG17
       test     r11d, r11d
       je       SHORT G_M000_IG17
       mov      r9, rbx
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xC0], r9
       mov      r9d, dword ptr [rbp+0x18]
       shl      r9, 2
       cmp      qword ptr [rbp-0xC0], r9
       jb       G_M000_IG88
       mov      r9d, r11d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xC0]
       jb       G_M000_IG88
 
G_M000_IG17:                ;; offset=0x0589
       test     ecx, ecx
       je       SHORT G_M000_IG18
       test     edi, edi
       je       SHORT G_M000_IG18
       mov      dword ptr [rbp-0x184], r8d
       mov      r9, bword ptr [rbp-0x1A8]
       mov      r8, r15
       sub      r8, r9
       mov      qword ptr [rbp-0xC8], r8
       mov      r8d, ecx
       shl      r8, 2
       cmp      qword ptr [rbp-0xC8], r8
       jb       G_M000_IG88
       mov      r8d, edi
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xC8]
       mov      bword ptr [rbp-0x1A8], r9
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG88
 
G_M000_IG18:                ;; offset=0x05E5
       test     edx, edx
       je       SHORT G_M000_IG19
       test     edi, edi
       je       SHORT G_M000_IG19
       mov      dword ptr [rbp-0x184], r8d
       mov      r9, bword ptr [rbp-0x1B0]
       mov      r8, r15
       sub      r8, r9
       mov      qword ptr [rbp-0xD0], r8
       mov      r8d, edx
       shl      r8, 2
       cmp      qword ptr [rbp-0xD0], r8
       jb       G_M000_IG88
       mov      r8d, edi
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xD0]
       mov      bword ptr [rbp-0x1B0], r9
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG88
 
G_M000_IG19:                ;; offset=0x0641
       test     r8d, r8d
       je       SHORT G_M000_IG20
       test     edi, edi
       je       SHORT G_M000_IG20
       mov      r9, r15
       sub      r9, qword ptr [rbp-0x1B8]
       mov      qword ptr [rbp-0xD8], r9
       mov      dword ptr [rbp-0x184], r8d
       mov      r9d, r8d
       shl      r9, 2
       cmp      qword ptr [rbp-0xD8], r9
       jb       G_M000_IG88
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xD8]
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG88
 
G_M000_IG20:                ;; offset=0x0694
       cmp      dword ptr [rbp+0x18], 0
       je       SHORT G_M000_IG21
       test     edi, edi
       je       SHORT G_M000_IG21
       mov      r9, r15
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xE0], r9
       mov      r9d, dword ptr [rbp+0x18]
       shl      r9, 2
       cmp      qword ptr [rbp-0xE0], r9
       jb       G_M000_IG88
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xE0]
       jb       G_M000_IG88
 
G_M000_IG21:                ;; offset=0x06D8
       test     ecx, ecx
       je       SHORT G_M000_IG22
       test     esi, esi
       je       SHORT G_M000_IG22
       mov      dword ptr [rbp-0x184], r8d
       mov      r9, bword ptr [rbp-0x1A8]
       mov      r8, r10
       sub      r8, r9
       mov      qword ptr [rbp-0xE8], r8
       mov      r8d, ecx
       shl      r8, 2
       cmp      qword ptr [rbp-0xE8], r8
       jb       G_M000_IG88
       mov      r8d, esi
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xE8]
       mov      bword ptr [rbp-0x1A8], r9
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG88
 
G_M000_IG22:                ;; offset=0x0734
       test     edx, edx
       je       SHORT G_M000_IG23
       test     esi, esi
       je       SHORT G_M000_IG23
       mov      dword ptr [rbp-0x184], r8d
       mov      r9, bword ptr [rbp-0x1B0]
       mov      r8, r10
       sub      r8, r9
       mov      qword ptr [rbp-0xF0], r8
       mov      r8d, edx
       shl      r8, 2
       cmp      qword ptr [rbp-0xF0], r8
       jb       G_M000_IG88
       mov      r8d, esi
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xF0]
       mov      bword ptr [rbp-0x1B0], r9
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG88
 
G_M000_IG23:                ;; offset=0x0790
       test     r8d, r8d
       je       SHORT G_M000_IG24
       test     esi, esi
       je       SHORT G_M000_IG24
       mov      r9, r10
       sub      r9, qword ptr [rbp-0x1B8]
       mov      qword ptr [rbp-0xF8], r9
       mov      dword ptr [rbp-0x184], r8d
       mov      r9d, r8d
       shl      r9, 2
       cmp      qword ptr [rbp-0xF8], r9
       jb       G_M000_IG88
       mov      r9d, esi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xF8]
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG88
 
G_M000_IG24:                ;; offset=0x07E3
       cmp      dword ptr [rbp+0x18], 0
       je       SHORT G_M000_IG25
       test     esi, esi
       je       SHORT G_M000_IG25
       mov      r9, r10
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0x100], r9
       mov      r9d, dword ptr [rbp+0x18]
       shl      r9, 2
       cmp      qword ptr [rbp-0x100], r9
       jb       G_M000_IG88
       mov      r9d, esi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x100]
       jb       G_M000_IG88
 
G_M000_IG25:                ;; offset=0x0827
       test     r11d, r11d
       je       G_M000_IG85
       test     edi, edi
       je       G_M000_IG85
       mov      r9, r15
       sub      r9, rbx
       mov      qword ptr [rbp-0x108], r9
       mov      r9d, r11d
       shl      r9, 2
       cmp      qword ptr [rbp-0x108], r9
       jb       G_M000_IG88
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x108]
       jb       G_M000_IG88
 
G_M000_IG26:                ;; offset=0x0870
       test     esi, esi
       je       G_M000_IG86
       mov      r9, r10
       sub      r9, rbx
       mov      qword ptr [rbp-0x110], r9
       mov      r9d, r11d
       shl      r9, 2
       cmp      qword ptr [rbp-0x110], r9
       jb       G_M000_IG88
       mov      r9d, esi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x110]
       jb       G_M000_IG88
 
G_M000_IG27:                ;; offset=0x08B0
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG28
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, rbx
       mov      qword ptr [rbp-0x118], r9
       mov      r9d, r11d
       shl      r9, 2
       cmp      qword ptr [rbp-0x118], r9
       jb       G_M000_IG88
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x118]
       jb       G_M000_IG88
 
G_M000_IG28:                ;; offset=0x08F0
       test     edi, edi
       je       G_M000_IG87
       test     esi, esi
       je       G_M000_IG87
       mov      r9, r10
       sub      r9, r15
       mov      qword ptr [rbp-0x120], r9
       mov      r9d, edi
       shl      r9, 2
       cmp      qword ptr [rbp-0x120], r9
       jb       G_M000_IG88
       mov      r9d, esi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x120]
       jb       G_M000_IG88
 
G_M000_IG29:                ;; offset=0x0938
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG30
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, r15
       mov      qword ptr [rbp-0x128], r9
       mov      dword ptr [rbp+0x48], edi
       mov      r9d, edi
       shl      r9, 2
       cmp      qword ptr [rbp-0x128], r9
       jb       G_M000_IG88
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x128]
       mov      edi, dword ptr [rbp+0x48]
       jb       G_M000_IG88
 
G_M000_IG30:                ;; offset=0x097E
       test     esi, esi
       je       G_M000_IG90
       cmp      dword ptr [rbp+0x28], 0
       je       G_M000_IG89
       mov      r9, bword ptr [rbp+0x20]
       mov      bword ptr [rbp+0x50], r10
       sub      r9, r10
       mov      qword ptr [rbp-0x130], r9
       mov      dword ptr [rbp+0x58], esi
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0x130], r9
       jb       G_M000_IG88
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x130]
       mov      dword ptr [rbp-0x184], r8d
       mov      esi, dword ptr [rbp+0x58]
       mov      r10, bword ptr [rbp+0x50]
       jb       G_M000_IG88
 
G_M000_IG31:                ;; offset=0x09DF
       mov      dword ptr [rbp-0x188], ecx
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r9, r9
       cmp      dword ptr [rbp-0x188], 0
       cmovne   r9, bword ptr [rbp-0x1A8]
       mov      bword ptr [rbp-0x138], r9
       mov      qword ptr [rbp-0x140], r9
       xor      r8d, r8d
       mov      dword ptr [rbp-0x17C], ecx
       cmp      ecx, 8
       jl       G_M000_IG91
       jmp      SHORT G_M000_IG33
       align    [0 bytes for IG32]
 
G_M000_IG32:                ;; offset=0x0A22
       mov      dword ptr [rbp-0x17C], ecx
 
G_M000_IG33:                ;; offset=0x0A28
       mov      r9d, r8d
       sar      r9d, 31
       and      r9d, 7
       add      r9d, r8d
       sar      r9d, 3
       movsxd   r9, r9d
       shl      r9, 5
       mov      rcx, qword ptr [rbp-0x140]
       vpand    ymm1, ymm0, ymmword ptr [r9+rcx]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG92
       add      r8d, 8
       lea      r9d, [r8+0x08]
       mov      ecx, dword ptr [rbp-0x17C]
       cmp      r9d, ecx
       jle      SHORT G_M000_IG32
 
G_M000_IG34:                ;; offset=0x0A70
       mov      dword ptr [rbp-0x17C], ecx
       cmp      r8d, ecx
       jl       G_M000_IG93
       xor      r8d, r8d
       mov      bword ptr [rbp-0x138], r8
       mov      r8d, 1
 
G_M000_IG35:                ;; offset=0x0A8F
       xor      r9, r9
       mov      bword ptr [rbp-0x138], r9
       test     r8d, r8d
       je       G_M000_IG111
       mov      r8d, edx
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r9, r9
       test     r8d, r8d
       cmovne   r9, bword ptr [rbp-0x1B0]
       mov      bword ptr [rbp-0x148], r9
       mov      qword ptr [rbp-0x150], r9
       xor      r8d, r8d
       cmp      edx, 8
       jl       G_M000_IG94
       jmp      SHORT G_M000_IG36
       align    [0 bytes for IG36]
 
G_M000_IG36:                ;; offset=0x0AD8
       mov      r9d, r8d
       sar      r9d, 31
       and      r9d, 7
       add      r9d, r8d
       sar      r9d, 3
       movsxd   r9, r9d
       shl      r9, 5
       mov      rcx, qword ptr [rbp-0x150]
       vpand    ymm1, ymm0, ymmword ptr [r9+rcx]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG95
       add      r8d, 8
       lea      r9d, [r8+0x08]
       cmp      r9d, edx
       jle      SHORT G_M000_IG36
 
G_M000_IG37:                ;; offset=0x0B1A
       mov      dword ptr [rbp-0x180], edx
       cmp      r8d, edx
       jl       G_M000_IG96
       xor      ecx, ecx
       mov      bword ptr [rbp-0x148], rcx
       mov      ecx, 1
 
G_M000_IG38:                ;; offset=0x0B37
       xor      r8, r8
       mov      bword ptr [rbp-0x148], r8
       test     ecx, ecx
       je       G_M000_IG111
       mov      r9d, dword ptr [rbp-0x184]
       mov      ecx, r9d
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r8, r8
       test     ecx, ecx
       cmovne   r8, bword ptr [rbp-0x1B8]
       mov      bword ptr [rbp-0x158], r8
       xor      ecx, ecx
       mov      dword ptr [rbp-0x184], r9d
       cmp      r9d, 8
       jl       SHORT G_M000_IG40
       align    [1 bytes for IG39]
 
G_M000_IG39:                ;; offset=0x0B80
       mov      r9d, ecx
       sar      r9d, 31
       and      r9d, 7
       add      r9d, ecx
       sar      r9d, 3
       movsxd   r9, r9d
       shl      r9, 5
       vpand    ymm1, ymm0, ymmword ptr [r9+r8]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG98
       add      ecx, 8
       lea      r9d, [rcx+0x08]
       cmp      r9d, dword ptr [rbp-0x184]
       jle      SHORT G_M000_IG39
 
G_M000_IG40:                ;; offset=0x0BBE
       cmp      ecx, dword ptr [rbp-0x184]
       jl       G_M000_IG99
       xor      ecx, ecx
       mov      bword ptr [rbp-0x158], rcx
       mov      ecx, 1
 
G_M000_IG41:                ;; offset=0x0BD8
       xor      r8, r8
       mov      bword ptr [rbp-0x158], r8
       test     ecx, ecx
       je       G_M000_IG111
       mov      r8d, dword ptr [rbp+0x18]
       mov      dword ptr [rbp-0x18C], r8d
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      rcx, rcx
       cmp      dword ptr [rbp-0x18C], 0
       cmovne   rcx, bword ptr [rbp+0x10]
       mov      bword ptr [rbp-0x160], rcx
       xor      r8d, r8d
       cmp      dword ptr [rbp+0x18], 8
       jl       SHORT G_M000_IG43
       align    [4 bytes for IG42]
 
G_M000_IG42:                ;; offset=0x0C20
       mov      r9d, r8d
       sar      r9d, 31
       and      r9d, 7
       add      r9d, r8d
       sar      r9d, 3
       movsxd   r9, r9d
       shl      r9, 5
       vpand    ymm1, ymm0, ymmword ptr [r9+rcx]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG100
       add      r8d, 8
       lea      r9d, [r8+0x08]
       cmp      r9d, dword ptr [rbp+0x18]
       jle      SHORT G_M000_IG42
 
G_M000_IG43:                ;; offset=0x0C5C
       cmp      r8d, dword ptr [rbp+0x18]
       jl       G_M000_IG101
       xor      ecx, ecx
       mov      bword ptr [rbp-0x160], rcx
       mov      ecx, 1
 
G_M000_IG44:                ;; offset=0x0C74
       xor      r8, r8
       mov      bword ptr [rbp-0x160], r8
       test     ecx, ecx
       je       G_M000_IG111
       lea      r8d, [r14+0x01]
       mov      dword ptr [rbp-0x19C], r8d
       mov      ecx, r8d
       shr      ecx, 31
       add      ecx, r8d
       sar      ecx, 1
       mov      dword ptr [rbp-0x44], ecx
       mov      r9d, dword ptr [rbp-0x198]
       shr      r9d, 31
       add      r9d, dword ptr [rbp-0x198]
       sar      r9d, 1
       imul     r9d, ecx
       jo       G_M000_IG113
       mov      dword ptr [rbp-0x48], r9d
       xor      edx, edx
       mov      eax, r12d
       imul     eax, r14d
       mov      dword ptr [rbp-0x194], eax
       cmp      edx, r9d
       jl       SHORT G_M000_IG51
       jmp      G_M000_IG102
 
G_M000_IG45:                ;; offset=0x0CDB
       mov      bword ptr [rbp+0x30], rbx
       mov      bword ptr [rbp+0x40], r15
       mov      r11d, dword ptr [rbp-0x4C]
 
G_M000_IG46:                ;; offset=0x0CE7
       xor      edi, edi
       mov      bword ptr [rbp-0x68], rdi
 
G_M000_IG47:                ;; offset=0x0CED
       mov      bword ptr [rbp-0x70], rdi
 
G_M000_IG48:                ;; offset=0x0CF1
       mov      bword ptr [rbp-0x78], rdi
 
G_M000_IG49:                ;; offset=0x0CF5
       mov      bword ptr [rbp-0x80], rdi
       add      r11d, 8
       mov      edi, dword ptr [rbp-0x48]
       cmp      r11d, edi
       mov      edx, r11d
       mov      r9d, edi
       mov      ecx, dword ptr [rbp-0x44]
       mov      rbx, bword ptr [rbp+0x30]
       mov      r15, bword ptr [rbp+0x40]
       jge      G_M000_IG79
 
G_M000_IG50:                ;; offset=0x0D1A
       mov      esi, dword ptr [rbp+0x58]
       mov      edi, dword ptr [rbp+0x48]
       mov      r10, bword ptr [rbp+0x50]
       mov      r11d, dword ptr [rbp+0x38]
 
G_M000_IG51:                ;; offset=0x0D28
       mov      eax, r9d
       sub      eax, edx
       cmp      eax, 8
       jl       G_M000_IG103
       mov      eax, 8
       mov      dword ptr [rbp+0x48], edi
       mov      dword ptr [rbp+0x58], esi
       mov      bword ptr [rbp+0x50], r10
 
G_M000_IG52:                ;; offset=0x0D45
       mov      dword ptr [rbp-0x50], eax
       mov      dword ptr [rsp], r14d
       mov      dword ptr [rsp+0x08], ecx
       mov      dword ptr [rbp-0x4C], edx
       mov      dword ptr [rsp+0x10], edx
       mov      dword ptr [rsp+0x18], eax
       mov      rdi, bword ptr [rbp-0x1A8]
       mov      esi, dword ptr [rbp-0x17C]
       mov      rdx, rbx
       mov      dword ptr [rbp+0x38], r11d
       mov      ecx, r11d
       mov      r8d, dword ptr [rbp+0x60]
       mov      r9d, r12d
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      eax, dword ptr [rbp+0x38]
       mov      edi, eax
       xor      rsi, rsi
       mov      bword ptr [rbp-0x168], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG53:                ;; offset=0x0D96
       test     edi, edi
       je       SHORT G_M000_IG54
       mov      rsi, rbx
 
G_M000_IG54:                ;; offset=0x0D9D
       mov      bword ptr [rbp-0x168], rsi
       xor      edi, edi
       cmp      eax, 8
       jl       SHORT G_M000_IG56
       align    [0 bytes for IG55]
 
G_M000_IG55:                ;; offset=0x0DAB
       mov      edx, edi
       sar      edx, 31
       and      edx, 7
       add      edx, edi
       sar      edx, 3
       movsxd   rdx, edx
       shl      rdx, 5
       vpand    ymm1, ymm0, ymmword ptr [rdx+rsi]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG104
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, eax
       jle      SHORT G_M000_IG55
       align    [0 bytes for IG56]
 
G_M000_IG56:                ;; offset=0x0DDD
       cmp      edi, eax
       jl       G_M000_IG105
       xor      edi, edi
       mov      bword ptr [rbp-0x168], rdi
       mov      edi, 1
 
G_M000_IG57:                ;; offset=0x0DF3
       xor      rsi, rsi
       mov      bword ptr [rbp-0x168], rsi
       test     edi, edi
       je       G_M000_IG111
       xor      rdi, rdi
       mov      dword ptr [rbp+0x38], eax
       test     eax, eax
       je       SHORT G_M000_IG58
       mov      bword ptr [rbp+0x30], rbx
       mov      rdi, rbx
       mov      rbx, bword ptr [rbp+0x30]
 
G_M000_IG58:                ;; offset=0x0E18
       mov      bword ptr [rbp-0x68], rdi
       xor      rsi, rsi
       cmp      dword ptr [rbp-0x180], 0
       je       SHORT G_M000_IG59
       mov      r10, bword ptr [rbp-0x1B0]
       mov      rsi, r10
 
G_M000_IG59:                ;; offset=0x0E31
       mov      bword ptr [rbp-0x70], rsi
       xor      r11, r11
       cmp      dword ptr [rbp+0x48], 0
       je       SHORT G_M000_IG60
       mov      r11, r15
 
G_M000_IG60:                ;; offset=0x0E41
       mov      bword ptr [rbp-0x78], r11
       mov      qword ptr [rbp-0x58], r11
       xor      rcx, rcx
       cmp      dword ptr [rbp+0x58], 0
       je       SHORT G_M000_IG61
       mov      rcx, bword ptr [rbp+0x50]
       mov      r10, rcx
       mov      bword ptr [rbp+0x50], rcx
       mov      rcx, r10
 
G_M000_IG61:                ;; offset=0x0E5F
       mov      bword ptr [rbp-0x80], rcx
       mov      qword ptr [rbp-0x60], rcx
       cmp      dword ptr [rbp+0x80], 16
       jne      G_M000_IG106
       mov      rdx, r11
       mov      ecx, dword ptr [rbp+0x60]
       mov      r8d, r13d
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int)]
 
G_M000_IG62:                ;; offset=0x0E83
       mov      r10d, dword ptr [rbp+0x48]
       mov      eax, r10d
       xor      rdx, rdx
       mov      bword ptr [rbp-0x170], rdx
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG63:                ;; offset=0x0E9C
       test     eax, eax
       je       SHORT G_M000_IG64
       mov      bword ptr [rbp+0x40], r15
       mov      rdx, r15
       mov      r15, bword ptr [rbp+0x40]
 
G_M000_IG64:                ;; offset=0x0EAB
       mov      bword ptr [rbp-0x170], rdx
       xor      eax, eax
       cmp      r10d, 8
       jl       SHORT G_M000_IG66
       align    [6 bytes for IG65]
 
G_M000_IG65:                ;; offset=0x0EC0
       mov      edi, eax
       sar      edi, 31
       and      edi, 7
       add      edi, eax
       sar      edi, 3
       movsxd   rdi, edi
       shl      rdi, 5
       vpand    ymm1, ymm0, ymmword ptr [rdi+rdx]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG107
       add      eax, 8
       lea      edi, [rax+0x08]
       cmp      edi, r10d
       jle      SHORT G_M000_IG65
       align    [0 bytes for IG66]
 
G_M000_IG66:                ;; offset=0x0EF3
       mov      dword ptr [rbp+0x48], r10d
       cmp      eax, r10d
       jl       G_M000_IG108
       xor      eax, eax
       mov      bword ptr [rbp-0x170], rax
       mov      eax, 1
 
G_M000_IG67:                ;; offset=0x0F0E
       xor      rdx, rdx
       mov      bword ptr [rbp-0x170], rdx
       test     eax, eax
       je       G_M000_IG111
       cmp      dword ptr [rbp+0x80], 16
       jne      G_M000_IG110
       mov      r9d, dword ptr [rbp-0x194]
       mov      edi, r9d
       xor      esi, esi
       cmp      esi, r13d
       jl       SHORT G_M000_IG71
       jmp      G_M000_IG45
 
G_M000_IG68:                ;; offset=0x0F42
       mov      edx, dword ptr [rbp-0x50]
       mov      rbx, qword ptr [rbp-0x60]
       mov      r11d, dword ptr [rbp-0x4C]
 
G_M000_IG69:                ;; offset=0x0F4D
       add      esi, 16
       cmp      esi, r13d
       jge      G_M000_IG46
 
G_M000_IG70:                ;; offset=0x0F59
       mov      dword ptr [rbp-0x4C], r11d
       mov      rbx, bword ptr [rbp+0x30]
       mov      r15, bword ptr [rbp+0x40]
 
G_M000_IG71:                ;; offset=0x0F65
       xor      ecx, ecx
       cmp      ecx, dword ptr [rbp-0x50]
       mov      bword ptr [rbp+0x30], rbx
       mov      bword ptr [rbp+0x40], r15
       jge      SHORT G_M000_IG68
       align    [0 bytes for IG72]
 
G_M000_IG72:                ;; offset=0x0F74
       lea      eax, [8*rsi]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x58]
       mov      r9d, ecx
       shl      r9d, 4
       movsxd   r9, r9d
       lea      r9, [rax+4*r9]
       lea      eax, [8*r13]
       mov      dword ptr [rbp-0x174], eax
       mov      r11d, dword ptr [rbp-0x4C]
       lea      r15d, [r11+rcx]
       mov      ebx, dword ptr [rbp-0x44]
       mov      eax, r15d
       cdq      
       idiv     edx:eax, ebx
       lea      edx, [rax+rax]
       mov      dword ptr [rbp-0x178], edx
       mov      eax, r15d
       cdq      
       idiv     edx:eax, ebx
       add      edx, edx
       vmovups  zmm0, zmmword ptr [r9]
       mov      r15d, dword ptr [rbp-0x174]
       lea      eax, [4*r15]
       cdqe     
       vmovups  zmm1, zmmword ptr [r9+4*rax]
       vaddps   zmm0, zmm0, zmm1
       lea      eax, [8*r15]
       cdqe     
       vmovups  zmm2, zmmword ptr [r9+4*rax]
       vaddps   zmm0, zmm0, zmm2
       vsubps   zmm1, zmm1, zmm2
       lea      eax, [r15+2*r15]
       lea      r8d, [4*rax]
       movsxd   r8, r8d
       vsubps   zmm1, zmm1, zmmword ptr [r9+4*r8]
       movsxd   r8, r15d
       vmovups  zmm2, zmmword ptr [r9+4*r8]
       lea      r8d, [r15+4*r15]
       movsxd   rbx, r8d
       vmovups  zmm3, zmmword ptr [r9+4*rbx]
       vaddps   zmm2, zmm2, zmm3
       lea      ebx, [r15+8*r15]
       movsxd   rbx, ebx
       vmovups  zmm4, zmmword ptr [r9+4*rbx]
       vaddps   zmm2, zmm2, zmm4
       vsubps   zmm3, zmm3, zmm4
       imul     ebx, r15d, 13
       movsxd   rbx, ebx
       vsubps   zmm3, zmm3, zmmword ptr [r9+4*rbx]
       lea      ebx, [r15+r15]
       movsxd   rbx, ebx
       vmovups  zmm4, zmmword ptr [r9+4*rbx]
       lea      ebx, [rax+rax]
       movsxd   rbx, ebx
       vmovups  zmm5, zmmword ptr [r9+4*rbx]
       vaddps   zmm4, zmm4, zmm5
       add      r8d, r8d
       movsxd   r8, r8d
       vmovups  zmm6, zmmword ptr [r9+4*r8]
       vaddps   zmm4, zmm6, zmm4
       vsubps   zmm5, zmm5, zmm6
       imul     r8d, r15d, 14
       movsxd   r8, r8d
       vsubps   zmm5, zmm5, zmmword ptr [r9+4*r8]
 
G_M000_IG73:                ;; offset=0x10A8
       movsxd   r8, eax
       vmovups  zmm6, zmmword ptr [r9+4*r8]
       lea      r8d, [8*r15]
       sub      r8d, r15d
       movsxd   r8, r8d
       vmovups  zmm7, zmmword ptr [r9+4*r8]
       vaddps   zmm6, zmm6, zmm7
       imul     r8d, r15d, 11
       movsxd   r8, r8d
       vmovups  zmm8, zmmword ptr [r9+4*r8]
       vaddps   zmm6, zmm6, zmm8
       vsubps   zmm7, zmm7, zmm8
       mov      r8d, r15d
       shl      r8d, 4
       sub      r8d, r15d
       movsxd   r8, r8d
       vsubps   zmm7, zmm7, zmmword ptr [r9+4*r8]
       vaddps   zmm0, zmm2, zmm0
       vaddps   zmm0, zmm0, zmm4
       vsubps   zmm2, zmm2, zmm4
       vsubps   zmm2, zmm2, zmm6
       vaddps   zmm1, zmm1, zmm3
       vaddps   zmm1, zmm1, zmm5
       vsubps   zmm3, zmm3, zmm5
       vsubps   zmm3, zmm3, zmm7
       mov      r8d, dword ptr [rbp-0x178]
       mov      r9d, r8d
       imul     r9d, r14d
       add      r9d, edx
       shl      r9d, 4
       movsxd   r9, r9d
       mov      eax, esi
       imul     eax, edi
       cdqe     
       shl      rax, 2
       mov      rbx, qword ptr [rbp-0x60]
       add      rax, rbx
       lea      r9, [rax+4*r9]
       vmovups  zmmword ptr [r9], zmm0
       lea      eax, [rdx+0x01]
       cmp      eax, r14d
       jge      SHORT G_M000_IG75
 
G_M000_IG74:                ;; offset=0x1167
       vmovups  zmmword ptr [r9+0x40], zmm2
 
G_M000_IG75:                ;; offset=0x116E
       inc      r8d
       cmp      r8d, r12d
       jge      SHORT G_M000_IG77
 
G_M000_IG76:                ;; offset=0x1176
       mov      r8d, r14d
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [r9+4*r8], zmm1
       inc      edx
       cmp      edx, r14d
       jge      SHORT G_M000_IG77
       mov      r15d, dword ptr [rbp-0x19C]
       mov      edx, r15d
       shl      edx, 4
       movsxd   rdx, edx
       vmovups  zmmword ptr [r9+4*rdx], zmm3
 
G_M000_IG77:                ;; offset=0x11A5
       inc      ecx
       mov      edx, dword ptr [rbp-0x50]
       cmp      ecx, edx
       jge      G_M000_IG69
 
G_M000_IG78:                ;; offset=0x11B2
       mov      dword ptr [rbp-0x4C], r11d
       jmp      G_M000_IG72
 
G_M000_IG79:                ;; offset=0x11BB
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG111
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG111
       mov      rdi, bword ptr [rbp-0x1B8]
       mov      esi, dword ptr [rbp-0x184]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG111
       mov      rdi, bword ptr [rbp+0x10]
       mov      esi, dword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG111
       mov      rbx, bword ptr [rbp+0x10]
       mov      bword ptr [rsp], rbx
       mov      r15d, dword ptr [rbp+0x18]
       mov      dword ptr [rsp+0x08], r15d
       mov      dword ptr [rsp+0x10], r13d
       mov      ebx, dword ptr [rbp-0x194]
       mov      dword ptr [rsp+0x18], ebx
       mov      ebx, dword ptr [rbp+0x80]
       mov      dword ptr [rsp+0x20], ebx
       movzx    rdx, byte  ptr [rbp+0x88]
       mov      dword ptr [rsp+0x28], edx
       mov      rdx, bword ptr [rbp+0x20]
       mov      ecx, dword ptr [rbp+0x28]
       mov      r8, bword ptr [rbp-0x1B8]
       mov      r9d, dword ptr [rbp-0x184]
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG80:                ;; offset=0x1271
       vzeroupper 
       add      rsp, 456
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG81:                ;; offset=0x1286
       mov      r9d, 1
       jmp      G_M000_IG04
 
G_M000_IG82:                ;; offset=0x1291
       mov      rdi, 0x7A8491C8CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0x101D8
       mov      rsi, 0x7A84924B9F30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG83:                ;; offset=0x12CD
       mov      rdi, 0x7A8491C8CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0x102B4
       mov      rsi, 0x7A84924B9F30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG84:                ;; offset=0x1309
       call     [System.ThrowHelper:ThrowArgumentOutOfRangeException()]
       int3     
 
G_M000_IG85:                ;; offset=0x1310
       test     r11d, r11d
       jne      G_M000_IG26
 
G_M000_IG86:                ;; offset=0x1319
       test     r11d, r11d
       je       G_M000_IG28
       jmp      G_M000_IG27
 
G_M000_IG87:                ;; offset=0x1327
       test     edi, edi
       je       G_M000_IG30
       jmp      G_M000_IG29
 
G_M000_IG88:                ;; offset=0x1334
       mov      rdi, 0x7A8491C8CD28
       call     CORINFO_HELP_NEWSFAST
       mov      r14, rax
       mov      edi, 0x102E2
       mov      rsi, 0x7A84924B9F30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, r14
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, r14
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG89:                ;; offset=0x1370
       mov      dword ptr [rbp-0x184], r8d
       jmp      G_M000_IG31
 
G_M000_IG90:                ;; offset=0x137C
       mov      dword ptr [rbp-0x184], r8d
       jmp      G_M000_IG31
 
G_M000_IG91:                ;; offset=0x1388
       mov      ecx, dword ptr [rbp-0x17C]
       jmp      G_M000_IG34
 
G_M000_IG92:                ;; offset=0x1393
       xor      r8d, r8d
       jmp      G_M000_IG35
 
G_M000_IG93:                ;; offset=0x139B
       movsxd   r9, r8d
       mov      rcx, qword ptr [rbp-0x140]
       mov      r9d, dword ptr [rcx+4*r9]
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, 0x7F800000
       mov      dword ptr [rbp-0x1C0], r9d
       mov      r9d, dword ptr [rbp-0x1BC]
       andn     r9d, r9d, dword ptr [rbp-0x1C0]
       je       SHORT G_M000_IG92
       inc      r8d
       mov      ecx, dword ptr [rbp-0x17C]
       jmp      G_M000_IG34
 
G_M000_IG94:                ;; offset=0x13DD
       mov      rcx, qword ptr [rbp-0x150]
       jmp      G_M000_IG37
 
G_M000_IG95:                ;; offset=0x13E9
       xor      ecx, ecx
       mov      dword ptr [rbp-0x180], edx
       jmp      G_M000_IG38
 
G_M000_IG96:                ;; offset=0x13F6
       movsxd   r9, r8d
       mov      r9d, dword ptr [rcx+4*r9]
       mov      dword ptr [rbp-0x1C0], r9d
       mov      r9d, 0x7F800000
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, dword ptr [rbp-0x1C0]
       andn     r9d, r9d, dword ptr [rbp-0x1BC]
       je       SHORT G_M000_IG97
       inc      r8d
       mov      edx, dword ptr [rbp-0x180]
       jmp      G_M000_IG37
 
G_M000_IG97:                ;; offset=0x1431
       mov      edx, dword ptr [rbp-0x180]
       jmp      SHORT G_M000_IG95
 
G_M000_IG98:                ;; offset=0x1439
       xor      ecx, ecx
       jmp      G_M000_IG41
 
G_M000_IG99:                ;; offset=0x1440
       movsxd   r9, ecx
       mov      r9d, dword ptr [r8+4*r9]
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, 0x7F800000
       mov      dword ptr [rbp-0x1C0], r9d
       mov      r9d, dword ptr [rbp-0x1BC]
       andn     r9d, r9d, dword ptr [rbp-0x1C0]
       je       SHORT G_M000_IG98
       inc      ecx
       jmp      G_M000_IG40
 
G_M000_IG100:                ;; offset=0x1474
       xor      ecx, ecx
       jmp      G_M000_IG44
 
G_M000_IG101:                ;; offset=0x147B
       movsxd   r9, r8d
       mov      r9d, dword ptr [rcx+4*r9]
       mov      dword ptr [rbp-0x1C0], r9d
       mov      r9d, 0x7F800000
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, dword ptr [rbp-0x1C0]
       andn     r9d, r9d, dword ptr [rbp-0x1BC]
       je       SHORT G_M000_IG100
       inc      r8d
       jmp      G_M000_IG43
 
G_M000_IG102:                ;; offset=0x14B0
       mov      dword ptr [rbp+0x58], esi
       mov      bword ptr [rbp+0x50], r10
       jmp      G_M000_IG79
 
G_M000_IG103:                ;; offset=0x14BC
       mov      dword ptr [rbp+0x48], edi
       mov      dword ptr [rbp+0x58], esi
       mov      bword ptr [rbp+0x50], r10
       jmp      G_M000_IG52
 
G_M000_IG104:                ;; offset=0x14CB
       xor      edi, edi
       jmp      G_M000_IG57
 
G_M000_IG105:                ;; offset=0x14D2
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      ecx, 0x7F800000
       andn     edx, edx, ecx
       je       SHORT G_M000_IG104
       inc      edi
       jmp      G_M000_IG56
 
G_M000_IG106:                ;; offset=0x14EB
       mov      rdx, r11
       mov      ecx, dword ptr [rbp+0x60]
       mov      r8d, r13d
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int)]
       jmp      G_M000_IG62
 
G_M000_IG107:                ;; offset=0x14FF
       xor      eax, eax
       mov      dword ptr [rbp+0x48], r10d
       jmp      G_M000_IG67
 
G_M000_IG108:                ;; offset=0x150A
       movsxd   rdi, eax
       mov      edi, dword ptr [rdx+4*rdi]
       mov      esi, 0x7F800000
       andn     edi, edi, esi
       je       SHORT G_M000_IG109
       inc      eax
       mov      r10d, dword ptr [rbp+0x48]
       jmp      G_M000_IG66
 
G_M000_IG109:                ;; offset=0x1527
       mov      r10d, dword ptr [rbp+0x48]
       jmp      SHORT G_M000_IG107
 
G_M000_IG110:                ;; offset=0x152D
       mov      eax, dword ptr [rbp-0x4C]
       mov      dword ptr [rsp], eax
       mov      edx, dword ptr [rbp-0x50]
       mov      dword ptr [rsp+0x08], edx
       mov      rdi, qword ptr [rbp-0x58]
       mov      rsi, qword ptr [rbp-0x60]
       mov      edx, r13d
       mov      ecx, r12d
       mov      r8d, r14d
       mov      r9d, dword ptr [rbp-0x44]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
       mov      bword ptr [rbp+0x30], rbx
       mov      bword ptr [rbp+0x40], r15
       mov      r11d, dword ptr [rbp-0x4C]
       jmp      G_M000_IG46
 
G_M000_IG111:                ;; offset=0x1566
       xor      eax, eax
 
G_M000_IG112:                ;; offset=0x1568
       vzeroupper 
       add      rsp, 456
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG113:                ;; offset=0x157D
       call     CORINFO_HELP_OVERFLOW
       int3     
 
RWD00  	dd	7F800000h		;       inf

; Total bytes of code 5507

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel512(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x1f7
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 1

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 64
       mov      qword ptr [rsp+0x7C8], r15
       mov      qword ptr [rsp+0x7C0], r14
       mov      qword ptr [rsp+0x7B8], r13
       mov      qword ptr [rsp+0x7B0], r12
       mov      qword ptr [rsp+0x7A8], rbx
       lea      rbp, [rsp+0x40]
       mov      rcx, qword ptr [rbp+0x760]
       mov      rdi, qword ptr [rbp+0x750]
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      eax, dword ptr [rbp+0x7A8]
       mov      r9d, dword ptr [rbp+0x7B8]
       mov      r12d, dword ptr [rbp+0x73C]
       mov      r13d, dword ptr [rbp+0x728]
       vmovups  zmm0, zmmword ptr [rbp+0x6E0]
       vmovups  zmm6, zmmword ptr [rbp+0x6A0]
       vmovups  zmm1, zmmword ptr [rbp+0x660]
       vmovups  zmm7, zmmword ptr [rbp+0x620]
       vmovups  zmm2, zmmword ptr [rbp+0x5E0]
       vmovups  zmm8, zmmword ptr [rbp+0x5A0]
       vmovups  zmm3, zmmword ptr [rbp+0x560]
       vmovups  zmm9, zmmword ptr [rbp+0x520]
       vmovups  zmm4, zmmword ptr [rbp+0x4E0]
       vmovups  zmm10, zmmword ptr [rbp+0x4A0]
       vmovups  zmm5, zmmword ptr [rbp+0x460]
       vmovups  zmm11, zmmword ptr [rbp+0x420]
       mov      rbx, qword ptr [rbp+0x418]
       mov      r15, qword ptr [rbp+0x410]
       mov      r10d, dword ptr [rbp+0x40C]
       mov      r14d, dword ptr [rbp+0x408]
       mov      r11d, dword ptr [rbp+0x404]
 
G_M000_IG02:                ;; offset=0x0106
       jmp      G_M000_IG10
 
G_M000_IG03:                ;; offset=0x010B
       inc      r10d
 
G_M000_IG04:                ;; offset=0x010E
       cmp      r10d, edx
       jge      G_M000_IG12
 
G_M000_IG05:                ;; offset=0x0117
       xor      r11d, r11d
       mov      r14d, r11d
       jmp      SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x011F
       inc      r14d
       cmp      r14d, 3
       jge      SHORT G_M000_IG03
 
G_M000_IG07:                ;; offset=0x0128
       xor      r11d, r11d
       mov      dword ptr [rbp+0x74C], edx
       mov      qword ptr [rbp+0x750], rdi
       mov      dword ptr [rbp+0x748], esi
       mov      dword ptr [rbp+0x7B8], r9d
 
G_M000_IG08:                ;; offset=0x0145
       vmovups  zmm12, zmmword ptr [rbx]
       vmovups  zmm13, zmmword ptr [r15]
       mov      r8d, r10d
       sar      r8d, 31
       and      r8d, 15
       add      r8d, r10d
       sar      r8d, 4
       mov      r9d, dword ptr [rbp+0x740]
       imul     r8d, r9d
       mov      esi, dword ptr [rbp+0x72C]
       mov      edi, esi
       imul     edi, eax
       add      edi, r8d
       add      edi, r14d
       imul     edi, r12d
       mov      r8d, r13d
       imul     r8d, eax
       add      edi, r8d
       add      edi, r11d
       shl      edi, 4
       movsxd   rdi, edi
       shl      rdi, 2
       add      rdi, rcx
       mov      r8d, r10d
       sar      r8d, 31
       and      r8d, 15
       add      r8d, r10d
       and      r8d, -16
       mov      edx, r10d
       sub      edx, r8d
       movsxd   rdx, edx
       lea      rdx, [rdi+4*rdx]
       vbroadcastss zmm14, dword ptr [rdx]
       vfmadd231ps zmm0, zmm12, zmm14
       vfmadd231ps zmm6, zmm13, zmm14
       mov      edi, eax
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm1, zmm12, zmm14
       vfmadd231ps zmm7, zmm13, zmm14
       lea      edi, [rax+rax]
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm2, zmm12, zmm14
       vfmadd231ps zmm8, zmm13, zmm14
       imul     edi, eax, 48
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm3, zmm12, zmm14
       vfmadd231ps zmm9, zmm13, zmm14
       lea      edi, [4*rax]
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm4, zmm12, zmm14
       vfmadd231ps zmm10, zmm13, zmm14
       imul     edi, eax, 80
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm5, zmm12, zmm14
       vfmadd231ps zmm11, zmm13, zmm14
       add      rbx, 64
       add      r15, 64
       inc      r11d
       mov      dword ptr [rbp+0x72C], esi
       mov      dword ptr [rbp+0x740], r9d
 
G_M000_IG09:                ;; offset=0x026F
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      rdi, qword ptr [rbp+0x750]
       mov      r9d, dword ptr [rbp+0x7B8]
 
G_M000_IG10:                ;; offset=0x0289
       cmp      r11d, 3
       jge      G_M000_IG06
 
G_M000_IG11:                ;; offset=0x0293
       mov      dword ptr [rbp+0x74C], edx
       mov      qword ptr [rbp+0x750], rdi
       mov      dword ptr [rbp+0x748], esi
       mov      dword ptr [rbp+0x7B8], r9d
       jmp      G_M000_IG08
 
G_M000_IG12:                ;; offset=0x02B2
       mov      r10d, dword ptr [rbp+0x730]
       mov      r11d, r10d
       sar      r11d, 31
       and      r11d, 15
       mov      dword ptr [rbp+0x730], r10d
       add      r11d, r10d
       sar      r11d, 4
       mov      r14d, dword ptr [rbp+0x738]
       mov      ebx, r11d
       imul     ebx, r14d
       mov      r15d, dword ptr [rbp+0x72C]
       mov      dword ptr [rbp+0x7B8], r9d
       mov      r8d, r15d
       imul     r8d, r9d
       add      ebx, r8d
       add      ebx, r13d
       mov      r10d, ebx
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm0
       mov      r10d, dword ptr [rbp+0x730]
       add      r10d, 16
       cmp      r10d, esi
       jge      SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x031C
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm6
 
G_M000_IG14:                ;; offset=0x0338
       lea      r9d, [rbx+0x01]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm1
       cmp      r10d, esi
       jge      SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x034F
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       lea      r9d, [r9+r13+0x01]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm7
 
G_M000_IG16:                ;; offset=0x036D
       lea      r9d, [rbx+0x02]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm2
       cmp      r10d, esi
       jge      SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x0384
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       lea      r9d, [r9+r13+0x02]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm8
 
G_M000_IG18:                ;; offset=0x03A2
       lea      r9d, [rbx+0x03]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm3
       cmp      r10d, esi
       jge      SHORT G_M000_IG20
 
G_M000_IG19:                ;; offset=0x03B9
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       lea      r9d, [r9+r13+0x03]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm9
 
G_M000_IG20:                ;; offset=0x03D7
       lea      r9d, [rbx+0x04]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm4
       cmp      r10d, esi
       jge      SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x03EE
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       lea      r9d, [r9+r13+0x04]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm10
 
G_M000_IG22:                ;; offset=0x040C
       add      ebx, 5
       shl      ebx, 4
       movsxd   r9, ebx
       vmovups  zmmword ptr [rdi+4*r9], zmm5
       mov      dword ptr [rbp+0x748], esi
       cmp      r10d, esi
       jge      SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x0427
       inc      r11d
       mov      dword ptr [rbp+0x738], r14d
       imul     r11d, r14d
       add      r8d, r11d
       lea      r8d, [r8+r13+0x05]
       shl      r8d, 4
       movsxd   r8, r8d
       mov      qword ptr [rbp+0x750], rdi
       vmovups  zmmword ptr [rdi+4*r8], zmm11
       mov      rdi, qword ptr [rbp+0x750]
       mov      r14d, dword ptr [rbp+0x738]
 
G_M000_IG24:                ;; offset=0x0460
       add      r13d, 6
 
G_M000_IG25:                ;; offset=0x0464
       lea      r8d, [r13+0x06]
       mov      r9d, dword ptr [rbp+0x7B8]
       cmp      r8d, r9d
       jg       G_M000_IG44
 
G_M000_IG26:                ;; offset=0x0478
       mov      r8d, r15d
       imul     r8d, r9d
       lea      r8d, [r8+r13+0x06]
       cmp      r8d, dword ptr [rbp+0x734]
       jg       G_M000_IG44
 
G_M000_IG27:                ;; offset=0x0491
       mov      dword ptr [rbp+0x7B8], r9d
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm7, ymm7, ymm7
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm8, ymm8, ymm8
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm9, ymm9, ymm9
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm10, ymm10, ymm10
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm11, ymm11, ymm11
       mov      ebx, dword ptr [rbp+0x730]
       mov      r11d, ebx
       imul     r11d, edx
       lea      r11d, [r11+8*r11]
       movsxd   r11, r11d
       mov      r10, qword ptr [rbp+0x758]
       lea      r11, [r10+4*r11]
       mov      dword ptr [rbp+0x74C], edx
       mov      r10d, edx
       shl      r10d, 4
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r11+4*r10]
       xor      r9d, r9d
       mov      dword ptr [rbp+0x72C], r15d
       mov      dword ptr [rbp+0x738], r14d
       mov      dword ptr [rbp+0x730], ebx
       mov      rbx, r11
       mov      r15, r10
       mov      r10d, r9d
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      r9d, dword ptr [rbp+0x7B8]
       jmp      G_M000_IG04
 
G_M000_IG28:                ;; offset=0x053B
       mov      rdi, qword ptr [rbp+0x750]
       inc      r9d
       cmp      r9d, edx
       jge      G_M000_IG41
 
G_M000_IG29:                ;; offset=0x054E
       mov      dword ptr [rbp+0x740], r8d
 
G_M000_IG30:                ;; offset=0x0555
       xor      r14d, r14d
       mov      esi, r9d
       sar      esi, 31
       and      esi, 15
       add      esi, r9d
       sar      esi, 4
       mov      r8d, dword ptr [rbp+0x740]
       imul     esi, r8d
       add      esi, dword ptr [rbp-0x30]
       mov      dword ptr [rbp-0x2C], esi
       mov      qword ptr [rbp+0x750], rdi
       jmp      G_M000_IG37
       align    [0 bytes for IG31]
 
G_M000_IG31:                ;; offset=0x0584
       vmulps   zmm7, zmm1, zmm7
       vaddps   zmm0, zmm7, zmm0
       vmulps   zmm1, zmm1, zmm2
       vaddps   zmm6, zmm1, zmm6
 
G_M000_IG32:                ;; offset=0x059C
       add      r11, 64
       add      r10, 64
       lea      edi, [rsi+0x01]
       shl      edi, 4
       add      edi, r9d
       movsxd   rdi, edi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [r11]
       vmovups  zmm2, zmmword ptr [r10]
       test     ebx, ebx
       je       G_M000_IG39
 
G_M000_IG33:                ;; offset=0x05CB
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
 
G_M000_IG34:                ;; offset=0x05D7
       add      r11, 64
       add      r10, 64
       add      esi, 2
       shl      esi, 4
       add      esi, r9d
       movsxd   rdi, esi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [r11]
       vmovups  zmm2, zmmword ptr [r10]
       test     ebx, ebx
       je       G_M000_IG40
 
G_M000_IG35:                ;; offset=0x0606
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
 
G_M000_IG36:                ;; offset=0x0612
       add      r11, 64
       add      r10, 64
       inc      r14d
       cmp      r14d, 3
       mov      esi, dword ptr [rbp-0x2C]
       mov      r9d, dword ptr [rbp+0x2C8]
       jge      G_M000_IG28
 
G_M000_IG37:                ;; offset=0x0631
       add      esi, r14d
       imul     esi, r12d
       add      esi, dword ptr [rbp-0x34]
       mov      edi, esi
       shl      edi, 4
       mov      ebx, r9d
       sar      ebx, 31
       and      ebx, 15
       add      ebx, r9d
       and      ebx, -16
       mov      dword ptr [rbp+0x2C8], r9d
       sub      r9d, ebx
       add      edi, r9d
       movsxd   rdi, edi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [r11]
       vmovups  zmm2, zmmword ptr [r10]
       mov      ebx, dword ptr [rbp+0x2CC]
       test     ebx, ebx
       je       G_M000_IG31
 
G_M000_IG38:                ;; offset=0x0680
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
       jmp      G_M000_IG32
 
G_M000_IG39:                ;; offset=0x0691
       vmulps   zmm7, zmm1, zmm7
       vaddps   zmm0, zmm7, zmm0
       vmulps   zmm1, zmm1, zmm2
       vaddps   zmm6, zmm1, zmm6
       jmp      G_M000_IG34
 
G_M000_IG40:                ;; offset=0x06AE
       vmulps   zmm8, zmm1, zmm7
       vaddps   zmm0, zmm8, zmm0
       vmulps   zmm3, zmm1, zmm2
       vaddps   zmm6, zmm3, zmm6
       jmp      G_M000_IG36
 
G_M000_IG41:                ;; offset=0x06CB
       mov      r10d, dword ptr [rbp+0x730]
       mov      esi, r10d
       sar      esi, 31
       and      esi, 15
       add      esi, r10d
       sar      esi, 4
       mov      r14d, dword ptr [rbp+0x738]
       mov      r9d, esi
       imul     r9d, r14d
       mov      r11d, dword ptr [rbp+0x7B8]
       mov      ebx, r15d
       imul     ebx, r11d
       add      r9d, ebx
       add      r9d, r13d
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rdi+4*r9], zmm0
       lea      r9d, [r10+0x10]
       mov      ebx, dword ptr [rbp+0x748]
       cmp      r9d, ebx
       jge      SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x0720
       inc      esi
       imul     esi, r14d
       mov      dword ptr [rbp+0x7B8], r11d
       mov      r9d, r15d
       imul     r9d, r11d
       add      esi, r9d
       add      esi, r13d
       shl      esi, 4
       movsxd   rsi, esi
       mov      qword ptr [rbp+0x750], rdi
       vmovups  zmmword ptr [rdi+4*rsi], zmm6
       mov      rdi, qword ptr [rbp+0x750]
       mov      r11d, dword ptr [rbp+0x7B8]
 
G_M000_IG43:                ;; offset=0x075C
       inc      r13d
       mov      dword ptr [rbp+0x740], r8d
       mov      dword ptr [rbp+0x748], ebx
       mov      dword ptr [rbp+0x730], r10d
       mov      r9d, r11d
 
G_M000_IG44:                ;; offset=0x0776
       cmp      r13d, r9d
       jge      G_M000_IG46
 
G_M000_IG45:                ;; offset=0x077F
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       mov      r11d, edx
       imul     r11d, dword ptr [rbp+0x730]
       lea      r11d, [r11+8*r11]
       movsxd   r11, r11d
       mov      rbx, qword ptr [rbp+0x758]
       lea      r11, [rbx+4*r11]
       mov      r10d, edx
       shl      r10d, 4
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r11+4*r10]
       mov      dword ptr [rbp+0x7B8], r9d
       mov      ebx, r15d
       imul     ebx, r9d
       add      ebx, r13d
       cmp      ebx, dword ptr [rbp+0x734]
       setl     bl
       movzx    rbx, bl
       mov      dword ptr [rbp+0x2CC], ebx
       xor      r9d, r9d
       mov      r8d, r15d
       imul     r8d, eax
       mov      dword ptr [rbp-0x30], r8d
       mov      r8d, r13d
       imul     r8d, eax
       mov      dword ptr [rbp-0x34], r8d
       cmp      r9d, edx
       mov      dword ptr [rbp+0x738], r14d
       jl       G_M000_IG30
       jmp      SHORT G_M000_IG50
 
G_M000_IG46:                ;; offset=0x0804
       inc      r15d
       mov      r11d, dword ptr [rbp+0x7B0]
       cmp      r15d, r11d
       jge      SHORT G_M000_IG51
 
G_M000_IG47:                ;; offset=0x0813
       xor      r13d, r13d
       mov      dword ptr [rbp+0x7B8], r9d
       mov      dword ptr [rbp+0x7B0], r11d
       jmp      G_M000_IG25
 
G_M000_IG48:                ;; offset=0x0829
       xor      r13d, r13d
       mov      dword ptr [rbp+0x7B0], r11d
       test     r11d, r11d
       mov      dword ptr [rbp+0x730], r15d
       mov      r11d, dword ptr [rbp+0x7B0]
       jle      SHORT G_M000_IG51
 
G_M000_IG49:                ;; offset=0x0846
       mov      r15d, r13d
       jmp      SHORT G_M000_IG47
 
G_M000_IG50:                ;; offset=0x084B
       mov      r8d, dword ptr [rbp+0x740]
       jmp      G_M000_IG41
 
G_M000_IG51:                ;; offset=0x0857
       mov      r15d, dword ptr [rbp+0x730]
       add      r15d, 32
       cmp      r15d, dword ptr [rbp+0x748]
       jl       SHORT G_M000_IG48
 
G_M000_IG52:                ;; offset=0x086B
       vzeroupper 
       add      rsp, 0x7A8
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 2176

