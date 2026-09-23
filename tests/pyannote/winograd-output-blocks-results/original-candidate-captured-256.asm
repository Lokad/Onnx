; Assembly listing for method KernelAccess:PrepareWinograd(System.ReadOnlySpan`1[float],int,int,int):float[] (Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       mov      rax, 0x76D2C9800190
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
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       mov      qword ptr [rbp-0x38], 0x1EA9B682
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
       mov      rdi, 0x76DAAF515DB8
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
       cmp      qword ptr [rbp-0x38], 0x1EA9B682
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
       cmp      qword ptr [rbp-0x38], 0x1EA9B682
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
       mov      rdi, 0x76DAAEC7CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0x10202
       mov      rsi, 0x76DAAF4A9F30
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
       mov      rdi, 0x76DAAF741E90
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
       mov      rdi, 0x76DAAF741E94
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
       mov      rdi, 0x76DAAF741E98
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
       mov      rdi, 0x76DAAF741E9C
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
       mov      rdi, 0x76DAAF741EA0
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
       mov      rdi, 0x76DAAF741EA4
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
       mov      rdi, 0x76DAAF741EA8
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
       mov      rdi, 0x76DAAF741EAC
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
       mov      rdi, 0x76DAAF741EB0
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
       mov      rdi, 0x76DAAF741EB4
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
       mov      rdi, 0x76DAAF741EB8
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
       mov      rdi, 0x76DAAF741EBC
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
       mov      rdi, 0x76DAAF741EC0
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
       mov      rdi, 0x76DAAF741EC4
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
       mov      rdi, 0x76DAAF741EC8
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
       mov      rdi, 0x76DAAF741ECC
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
       mov      rdi, 0x76DAAF741ED0
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
       mov      rdi, 0x76DAAF741ED4
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
       mov      rdi, 0x76DAAF741ED8
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
       mov      rdi, 0x76DAAF741EDC
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG54:                ;; offset=0x0DB5
       vzeroupper 
       add      rsp, 0x400
       pop      rbp
       ret      
 
; Total bytes of code 3521

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel256(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x1ec
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 1

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 64
       mov      qword ptr [rsp+0x448], r15
       mov      qword ptr [rsp+0x440], r14
       mov      qword ptr [rsp+0x438], r13
       mov      qword ptr [rsp+0x430], r12
       mov      qword ptr [rsp+0x428], rbx
       lea      rbp, [rsp+0x40]
       mov      rcx, qword ptr [rbp+0x3E0]
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      eax, dword ptr [rbp+0x428]
       mov      r8d, dword ptr [rbp+0x438]
       mov      r12d, dword ptr [rbp+0x3BC]
       mov      r13d, dword ptr [rbp+0x3A8]
       vmovups  ymm0, ymmword ptr [rbp+0x380]
       vmovups  ymm6, ymmword ptr [rbp+0x360]
       vmovups  ymm1, ymmword ptr [rbp+0x340]
       vmovups  ymm7, ymmword ptr [rbp+0x320]
       vmovups  ymm2, ymmword ptr [rbp+0x300]
       vmovups  ymm8, ymmword ptr [rbp+0x2E0]
       vmovups  ymm3, ymmword ptr [rbp+0x2C0]
       vmovups  ymm9, ymmword ptr [rbp+0x2A0]
       vmovups  ymm4, ymmword ptr [rbp+0x280]
       vmovups  ymm10, ymmword ptr [rbp+0x260]
       vmovups  ymm5, ymmword ptr [rbp+0x240]
       vmovups  ymm11, ymmword ptr [rbp+0x220]
       mov      rbx, qword ptr [rbp+0x218]
       mov      r15, qword ptr [rbp+0x210]
       mov      r10d, dword ptr [rbp+0x20C]
       mov      r14d, dword ptr [rbp+0x208]
       mov      r11d, dword ptr [rbp+0x204]
 
G_M000_IG02:                ;; offset=0x00EE
       jmp      G_M000_IG10
 
G_M000_IG03:                ;; offset=0x00F3
       inc      r10d
 
G_M000_IG04:                ;; offset=0x00F6
       cmp      r10d, edx
       jge      G_M000_IG12
 
G_M000_IG05:                ;; offset=0x00FF
       xor      r9d, r9d
       mov      r14d, r9d
       jmp      SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x0107
       inc      r14d
       cmp      r14d, 3
       jge      SHORT G_M000_IG03
 
G_M000_IG07:                ;; offset=0x0110
       xor      r9d, r9d
       mov      dword ptr [rbp+0x3CC], edx
       mov      qword ptr [rbp+0x3D0], rdi
       mov      dword ptr [rbp+0x3C8], esi
       mov      dword ptr [rbp+0x438], r8d
       mov      r11d, r9d
 
G_M000_IG08:                ;; offset=0x0130
       vmovups  ymm12, ymmword ptr [rbx]
       vmovups  ymm13, ymmword ptr [r15]
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 7
       add      r9d, r10d
       sar      r9d, 3
       mov      r8d, dword ptr [rbp+0x3C0]
       imul     r9d, r8d
       mov      esi, dword ptr [rbp+0x3AC]
       mov      edi, esi
       imul     edi, eax
       add      edi, r9d
       add      edi, r14d
       imul     edi, r12d
       mov      r9d, r13d
       imul     r9d, eax
       add      edi, r9d
       add      edi, r11d
       shl      edi, 3
       movsxd   rdi, edi
       shl      rdi, 2
       add      rdi, rcx
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 7
       add      r9d, r10d
       and      r9d, -8
       mov      edx, r10d
       sub      edx, r9d
       movsxd   rdx, edx
       lea      rdx, [rdi+4*rdx]
       vbroadcastss ymm14, dword ptr [rdx]
       vfmadd231ps ymm0, ymm12, ymm14
       vfmadd231ps ymm6, ymm13, ymm14
       lea      edi, [8*rax]
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm1, ymm12, ymm14
       vfmadd231ps ymm7, ymm13, ymm14
       lea      edi, [rax+rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm2, ymm12, ymm14
       vfmadd231ps ymm8, ymm13, ymm14
       lea      edi, [rax+2*rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm3, ymm12, ymm14
       vfmadd231ps ymm9, ymm13, ymm14
       lea      edi, [4*rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm4, ymm12, ymm14
       vfmadd231ps ymm10, ymm13, ymm14
       lea      edi, [rax+4*rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm5, ymm12, ymm14
       vfmadd231ps ymm11, ymm13, ymm14
       add      rbx, 32
       add      r15, 32
       inc      r11d
       mov      dword ptr [rbp+0x3AC], esi
       mov      dword ptr [rbp+0x3C0], r8d
 
G_M000_IG09:                ;; offset=0x024D
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r8d, dword ptr [rbp+0x438]
 
G_M000_IG10:                ;; offset=0x0267
       cmp      r11d, 3
       jge      G_M000_IG06
 
G_M000_IG11:                ;; offset=0x0271
       mov      dword ptr [rbp+0x3CC], edx
       mov      qword ptr [rbp+0x3D0], rdi
       mov      dword ptr [rbp+0x3C8], esi
       mov      dword ptr [rbp+0x438], r8d
       jmp      G_M000_IG08
 
G_M000_IG12:                ;; offset=0x0290
       mov      r10d, dword ptr [rbp+0x3B0]
       mov      r11d, r10d
       sar      r11d, 31
       and      r11d, 7
       add      r11d, r10d
       sar      r11d, 3
       mov      r14d, dword ptr [rbp+0x3B8]
       mov      ebx, r11d
       imul     ebx, r14d
       mov      r15d, dword ptr [rbp+0x3AC]
       mov      dword ptr [rbp+0x438], r8d
       mov      r9d, r15d
       imul     r9d, r8d
       add      ebx, r9d
       add      ebx, r13d
       shl      ebx, 3
       movsxd   r8, ebx
       vmovups  ymmword ptr [rdi+4*r8], ymm0
       mov      dword ptr [rbp+0x3B0], r10d
       lea      r8d, [r10+0x08]
       cmp      r8d, esi
       jge      SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x02EE
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       shl      r10d, 3
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm6
 
G_M000_IG14:                ;; offset=0x0309
       lea      r10d, [rbx+0x08]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm1
       cmp      r8d, esi
       jge      SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x031B
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       lea      r10d, [8*r10+0x08]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm7
 
G_M000_IG16:                ;; offset=0x033A
       lea      r10d, [rbx+0x10]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm2
       cmp      r8d, esi
       jge      SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x034C
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       lea      r10d, [8*r10+0x10]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm8
 
G_M000_IG18:                ;; offset=0x036B
       lea      r10d, [rbx+0x18]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm3
       cmp      r8d, esi
       jge      SHORT G_M000_IG20
 
G_M000_IG19:                ;; offset=0x037D
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       lea      r10d, [8*r10+0x18]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm9
 
G_M000_IG20:                ;; offset=0x039C
       lea      r10d, [rbx+0x20]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm4
       cmp      r8d, esi
       jge      SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x03AE
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       lea      r10d, [8*r10+0x20]
       movsxd   r10, r10d
       vmovups  ymmword ptr [rdi+4*r10], ymm10
 
G_M000_IG22:                ;; offset=0x03CD
       add      ebx, 40
       movsxd   r10, ebx
       vmovups  ymmword ptr [rdi+4*r10], ymm5
       mov      dword ptr [rbp+0x3C8], esi
       cmp      r8d, esi
       jge      SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x03E4
       inc      r11d
       mov      dword ptr [rbp+0x3B8], r14d
       imul     r11d, r14d
       add      r9d, r11d
       add      r9d, r13d
       lea      r8d, [8*r9+0x28]
       movsxd   r8, r8d
       mov      qword ptr [rbp+0x3D0], rdi
       vmovups  ymmword ptr [rdi+4*r8], ymm11
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r14d, dword ptr [rbp+0x3B8]
 
G_M000_IG24:                ;; offset=0x041E
       add      r13d, 6
 
G_M000_IG25:                ;; offset=0x0422
       lea      r8d, [r13+0x06]
       mov      r9d, dword ptr [rbp+0x438]
       cmp      r8d, r9d
       jg       G_M000_IG28
 
G_M000_IG26:                ;; offset=0x0436
       mov      r8d, r15d
       imul     r8d, r9d
       lea      r8d, [r8+r13+0x06]
       cmp      r8d, dword ptr [rbp+0x3B4]
       jg       G_M000_IG28
 
G_M000_IG27:                ;; offset=0x044F
       mov      dword ptr [rbp+0x438], r9d
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
       mov      ebx, dword ptr [rbp+0x3B0]
       mov      r8d, ebx
       imul     r8d, edx
       lea      r8d, [r8+8*r8]
       movsxd   r8, r8d
       mov      r10, qword ptr [rbp+0x3D8]
       lea      r8, [r10+4*r8]
       mov      dword ptr [rbp+0x3CC], edx
       lea      r10d, [8*rdx]
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r8+4*r10]
       xor      r9d, r9d
       mov      dword ptr [rbp+0x3AC], r15d
       mov      dword ptr [rbp+0x3B8], r14d
       mov      dword ptr [rbp+0x3B0], ebx
       mov      rbx, r8
       mov      r15, r10
       mov      r10d, r9d
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      r8d, dword ptr [rbp+0x438]
       jmp      G_M000_IG04
 
G_M000_IG28:                ;; offset=0x04FA
       mov      r8d, r15d
       imul     r8d, eax
       mov      dword ptr [rbp-0x30], r8d
       jmp      G_M000_IG35
       align    [0 bytes for IG29]
 
G_M000_IG29:                ;; offset=0x050A
       vmulps   ymm7, ymm1, ymm7
       vaddps   ymm0, ymm7, ymm0
       vmulps   ymm1, ymm1, ymm2
       vaddps   ymm6, ymm1, ymm6
       jmp      G_M000_IG43
 
G_M000_IG30:                ;; offset=0x051F
       vmulps   ymm8, ymm1, ymm7
       vaddps   ymm0, ymm8, ymm0
       vmulps   ymm3, ymm1, ymm2
       vaddps   ymm6, ymm3, ymm6
       jmp      G_M000_IG45
 
G_M000_IG31:                ;; offset=0x0534
       mov      r11d, dword ptr [rbp+0x3C0]
 
G_M000_IG32:                ;; offset=0x053B
       mov      r10d, dword ptr [rbp+0x3B0]
       mov      esi, r10d
       sar      esi, 31
       and      esi, 7
       add      esi, r10d
       sar      esi, 3
       mov      r14d, dword ptr [rbp+0x3B8]
       mov      r8d, esi
       imul     r8d, r14d
       mov      r9d, dword ptr [rbp+0x438]
       mov      ebx, r15d
       imul     ebx, r9d
       add      r8d, ebx
       add      r8d, r13d
       shl      r8d, 3
       movsxd   r8, r8d
       vmovups  ymmword ptr [rdi+4*r8], ymm0
       lea      r8d, [r10+0x08]
       mov      ebx, dword ptr [rbp+0x3C8]
       cmp      r8d, ebx
       jge      SHORT G_M000_IG34
 
G_M000_IG33:                ;; offset=0x058F
       inc      esi
       imul     esi, r14d
       mov      dword ptr [rbp+0x438], r9d
       mov      r8d, r15d
       imul     r8d, r9d
       add      esi, r8d
       add      esi, r13d
       shl      esi, 3
       movsxd   rsi, esi
       mov      qword ptr [rbp+0x3D0], rdi
       vmovups  ymmword ptr [rdi+4*rsi], ymm6
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r9d, dword ptr [rbp+0x438]
 
G_M000_IG34:                ;; offset=0x05C9
       inc      r13d
       mov      dword ptr [rbp+0x3C0], r11d
       mov      dword ptr [rbp+0x3C8], ebx
       mov      dword ptr [rbp+0x3B0], r10d
 
G_M000_IG35:                ;; offset=0x05E0
       cmp      r13d, r9d
       jge      G_M000_IG48
 
G_M000_IG36:                ;; offset=0x05E9
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       mov      ebx, edx
       imul     ebx, dword ptr [rbp+0x3B0]
       lea      ebx, [rbx+8*rbx]
       movsxd   rbx, ebx
       mov      r10, qword ptr [rbp+0x3D8]
       lea      rbx, [r10+4*rbx]
       lea      r8d, [8*rdx]
       lea      r8d, [r8+8*r8]
       movsxd   r8, r8d
       lea      r8, [rbx+4*r8]
       mov      dword ptr [rbp+0x438], r9d
       mov      r10d, r15d
       imul     r10d, r9d
       add      r10d, r13d
       cmp      r10d, dword ptr [rbp+0x3B4]
       setl     r10b
       movzx    r10, r10b
       mov      dword ptr [rbp+0x16C], r10d
       xor      r9d, r9d
       mov      r11d, r13d
       imul     r11d, eax
       mov      dword ptr [rbp-0x34], r11d
       cmp      r9d, edx
       mov      dword ptr [rbp+0x3B8], r14d
       jl       SHORT G_M000_IG39
       jmp      G_M000_IG31
 
G_M000_IG37:                ;; offset=0x0664
       mov      rdi, qword ptr [rbp+0x3D0]
       inc      r9d
       cmp      r9d, edx
       jge      G_M000_IG32
 
G_M000_IG38:                ;; offset=0x0677
       mov      dword ptr [rbp+0x3C0], r11d
 
G_M000_IG39:                ;; offset=0x067E
       xor      r14d, r14d
       mov      esi, r9d
       sar      esi, 31
       and      esi, 7
       add      esi, r9d
       sar      esi, 3
       mov      r11d, dword ptr [rbp+0x3C0]
       imul     esi, r11d
       add      esi, dword ptr [rbp-0x30]
       mov      dword ptr [rbp-0x2C], esi
       mov      qword ptr [rbp+0x3D0], rdi
       jmp      G_M000_IG46
 
G_M000_IG40:                ;; offset=0x06AD
       vmulps   ymm7, ymm1, ymm7
       vaddps   ymm0, ymm7, ymm0
       vmulps   ymm1, ymm1, ymm2
       vaddps   ymm6, ymm1, ymm6
 
G_M000_IG41:                ;; offset=0x06BD
       add      rbx, 32
       add      r8, 32
       lea      edi, [rsi+0x01]
       lea      edi, [r10+8*rdi]
       movsxd   rdi, edi
       vbroadcastss ymm1, dword ptr [rcx+4*rdi]
       vmovups  ymm7, ymmword ptr [rbx]
       vmovups  ymm2, ymmword ptr [r8]
       mov      edi, dword ptr [rbp+0x16C]
       test     edi, edi
       je       G_M000_IG29
 
G_M000_IG42:                ;; offset=0x06EC
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
 
G_M000_IG43:                ;; offset=0x06F6
       add      rbx, 32
       add      r8, 32
       add      esi, 2
       lea      esi, [r10+8*rsi]
       movsxd   rsi, esi
       vbroadcastss ymm1, dword ptr [rcx+4*rsi]
       vmovups  ymm7, ymmword ptr [rbx]
       vmovups  ymm2, ymmword ptr [r8]
       test     edi, edi
       je       G_M000_IG30
 
G_M000_IG44:                ;; offset=0x071F
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
 
G_M000_IG45:                ;; offset=0x0729
       add      rbx, 32
       add      r8, 32
       inc      r14d
       cmp      r14d, 3
       mov      esi, dword ptr [rbp-0x2C]
       jge      G_M000_IG37
 
G_M000_IG46:                ;; offset=0x0741
       add      esi, r14d
       imul     esi, r12d
       add      esi, dword ptr [rbp-0x34]
       mov      edi, r9d
       sar      edi, 31
       and      edi, 7
       add      edi, r9d
       and      edi, -8
       mov      r10d, r9d
       sub      r10d, edi
       lea      edi, [r10+8*rsi]
       movsxd   rdi, edi
       vbroadcastss ymm1, dword ptr [rcx+4*rdi]
       vmovups  ymm7, ymmword ptr [rbx]
       vmovups  ymm2, ymmword ptr [r8]
       cmp      dword ptr [rbp+0x16C], 0
       je       G_M000_IG40
 
G_M000_IG47:                ;; offset=0x0783
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
       jmp      G_M000_IG41
 
G_M000_IG48:                ;; offset=0x0792
       inc      r15d
       mov      r8d, dword ptr [rbp+0x430]
       cmp      r15d, r8d
       jge      SHORT G_M000_IG52
 
G_M000_IG49:                ;; offset=0x07A1
       xor      r13d, r13d
       mov      dword ptr [rbp+0x438], r9d
       mov      dword ptr [rbp+0x430], r8d
       jmp      G_M000_IG25
 
G_M000_IG50:                ;; offset=0x07B7
       xor      r13d, r13d
       mov      dword ptr [rbp+0x430], r8d
       test     r8d, r8d
       mov      dword ptr [rbp+0x3B0], r15d
       mov      r8d, dword ptr [rbp+0x430]
       jle      SHORT G_M000_IG52
 
G_M000_IG51:                ;; offset=0x07D4
       mov      r15d, r13d
       jmp      SHORT G_M000_IG49
 
G_M000_IG52:                ;; offset=0x07D9
       mov      r15d, dword ptr [rbp+0x3B0]
       add      r15d, 16
       cmp      r15d, dword ptr [rbp+0x3C8]
       jl       SHORT G_M000_IG50
 
G_M000_IG53:                ;; offset=0x07ED
       vzeroupper 
       add      rsp, 0x428
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 2050

; Assembly listing for method KernelAccess:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       mov      rax, 0x76D2C9800198
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
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       mov      rax, 0x76D2C98001A8
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
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; partially interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x490
       lea      rbp, [rsp+0x490]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x430], xmm8
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
 
G_M000_IG02:                ;; offset=0x005B
       mov      dword ptr [rbp-0x420], 0x3E8
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
       mov      rdi, 0x76DAAF74AE60
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG97
 
G_M000_IG03:                ;; offset=0x00BE
       mov      eax, dword ptr [rbp+0x60]
       imul     eax, dword ptr [rbp+0x70]
       jo       G_M000_IG103
       imul     eax, dword ptr [rbp+0x78]
       jo       G_M000_IG103
       cmp      dword ptr [rbp-0x30], eax
       jne      G_M000_IG07
       imul     eax, dword ptr [rbp+0x60], 16
       jo       G_M000_IG103
       imul     eax, dword ptr [rbp+0x68]
       jo       G_M000_IG103
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
       mov      rdi, 0x76DAAF74AE64
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG04:                ;; offset=0x0128
       cmp      dword ptr [rbp+0x18], 0
       je       SHORT G_M000_IG05
       mov      eax, dword ptr [rbp+0x18]
       cmp      eax, dword ptr [rbp-0x70]
       jne      G_M000_IG10
       mov      rdi, 0x76DAAF74AE68
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG05:                ;; offset=0x0149
       mov      eax, dword ptr [rbp+0x38]
       cmp      eax, dword ptr [rbp-0x60]
       jl       G_M000_IG09
       mov      eax, dword ptr [rbp+0x48]
       cmp      eax, dword ptr [rbp-0x68]
       jl       SHORT G_M000_IG08
       mov      eax, dword ptr [rbp+0x58]
       cmp      eax, dword ptr [rbp-0x70]
       jge      G_M000_IG14
 
G_M000_IG06:                ;; offset=0x0169
       mov      rdi, 0x76DAAF74AE6C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG07:                ;; offset=0x0178
       mov      rdi, 0x76DAAEC7CD28
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0xC8], rax
       mov      edi, 0x102B4
       mov      rsi, 0x76DAAF4A9F30
       call     [CORINFO_HELP_STRCNS]
       mov      gword ptr [rbp-0x428], rax
       mov      rsi, gword ptr [rbp-0x428]
       mov      rdi, gword ptr [rbp-0xC8]
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, gword ptr [rbp-0xC8]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG08:                ;; offset=0x01CB
       mov      rdi, 0x76DAAF74AE70
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG07
 
G_M000_IG09:                ;; offset=0x01DC
       mov      rdi, 0x76DAAF74AE74
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG07
 
G_M000_IG10:                ;; offset=0x01ED
       mov      rdi, 0x76DAAF74AE78
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG11:                ;; offset=0x0201
       mov      rdi, 0x76DAAF74AE7C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG12:                ;; offset=0x0215
       mov      rdi, 0x76DAAF74AE80
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG13:                ;; offset=0x0229
       mov      rdi, 0x76DAAF74AE84
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG14:                ;; offset=0x023D
       lea      rdi, [rbp+0x30]
       mov      edx, dword ptr [rbp-0x60]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xD8], rax
       mov      qword ptr [rbp-0xD0], rdx
 
G_M000_IG15:                ;; offset=0x025A
       vmovdqu  xmm0, xmmword ptr [rbp-0xD8]
       vmovdqu  xmmword ptr [rbp+0x30], xmm0
 
G_M000_IG16:                ;; offset=0x0267
       lea      rdi, [rbp+0x40]
       mov      edx, dword ptr [rbp-0x68]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xE8], rax
       mov      qword ptr [rbp-0xE0], rdx
 
G_M000_IG17:                ;; offset=0x0284
       vmovdqu  xmm0, xmmword ptr [rbp-0xE8]
       vmovdqu  xmmword ptr [rbp+0x40], xmm0
 
G_M000_IG18:                ;; offset=0x0291
       lea      rdi, [rbp+0x50]
       mov      edx, dword ptr [rbp-0x70]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xF8], rax
       mov      qword ptr [rbp-0xF0], rdx
 
G_M000_IG19:                ;; offset=0x02AE
       vmovdqu  xmm0, xmmword ptr [rbp-0xF8]
       vmovdqu  xmmword ptr [rbp+0x50], xmm0
 
G_M000_IG20:                ;; offset=0x02BB
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu  xmmword ptr [rbp-0x118], xmm0
 
G_M000_IG21:                ;; offset=0x02C8
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
 
G_M000_IG22:                ;; offset=0x030E
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu  xmmword ptr [rbp-0x140], xmm0
 
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
       vmovdqu  xmmword ptr [rbp-0x160], xmm0
 
G_M000_IG25:                ;; offset=0x036E
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
 
G_M000_IG26:                ;; offset=0x03B4
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu  xmmword ptr [rbp-0x180], xmm0
 
G_M000_IG27:                ;; offset=0x03C1
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
 
G_M000_IG28:                ;; offset=0x0407
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu  xmmword ptr [rbp-0x1A0], xmm0
 
G_M000_IG29:                ;; offset=0x0414
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
 
G_M000_IG30:                ;; offset=0x045A
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu  xmmword ptr [rbp-0x1C0], xmm0
 
G_M000_IG31:                ;; offset=0x0467
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
 
G_M000_IG32:                ;; offset=0x04AD
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu  xmmword ptr [rbp-0x1E0], xmm0
 
G_M000_IG33:                ;; offset=0x04BA
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
 
G_M000_IG34:                ;; offset=0x0500
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu  xmmword ptr [rbp-0x200], xmm0
 
G_M000_IG35:                ;; offset=0x050D
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
 
G_M000_IG36:                ;; offset=0x0553
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu  xmmword ptr [rbp-0x220], xmm0
 
G_M000_IG37:                ;; offset=0x0560
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
 
G_M000_IG38:                ;; offset=0x05A6
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu  xmmword ptr [rbp-0x240], xmm0
 
G_M000_IG39:                ;; offset=0x05B3
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
 
G_M000_IG40:                ;; offset=0x05F9
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu  xmmword ptr [rbp-0x260], xmm0
 
G_M000_IG41:                ;; offset=0x0606
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
 
G_M000_IG42:                ;; offset=0x064C
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu  xmmword ptr [rbp-0x280], xmm0
 
G_M000_IG43:                ;; offset=0x0659
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
 
G_M000_IG44:                ;; offset=0x069F
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu  xmmword ptr [rbp-0x2A0], xmm0
 
G_M000_IG45:                ;; offset=0x06AC
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
 
G_M000_IG46:                ;; offset=0x06F2
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu  xmmword ptr [rbp-0x2C0], xmm0
 
G_M000_IG47:                ;; offset=0x06FF
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
 
G_M000_IG48:                ;; offset=0x0745
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu  xmmword ptr [rbp-0x2E0], xmm0
 
G_M000_IG49:                ;; offset=0x0752
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
 
G_M000_IG50:                ;; offset=0x0798
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu  xmmword ptr [rbp-0x300], xmm0
 
G_M000_IG51:                ;; offset=0x07A5
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
 
G_M000_IG52:                ;; offset=0x08E7
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
 
G_M000_IG53:                ;; offset=0x0A2F
       test     eax, eax
       je       G_M000_IG76
 
G_M000_IG54:                ;; offset=0x0A37
       mov      rdi, 0x76DAAF74AE88
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG55:                ;; offset=0x0A46
       mov      rdi, 0x76DAAEC7CD28
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0x120], rax
       mov      edi, 0x102E2
       mov      rsi, 0x76DAAF4A9F30
       call     [CORINFO_HELP_STRCNS]
       mov      gword ptr [rbp-0x430], rax
       mov      rsi, gword ptr [rbp-0x430]
       mov      rdi, gword ptr [rbp-0x120]
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, gword ptr [rbp-0x120]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG56:                ;; offset=0x0A99
       mov      rdi, 0x76DAAF74AE8C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG55
 
G_M000_IG57:                ;; offset=0x0AAA
       mov      rdi, 0x76DAAF74AE90
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG55
 
G_M000_IG58:                ;; offset=0x0ABB
       mov      rdi, 0x76DAAF74AE94
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG59:                ;; offset=0x0ACF
       mov      rdi, 0x76DAAF74AE98
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG60:                ;; offset=0x0AE3
       mov      rdi, 0x76DAAF74AE9C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG61:                ;; offset=0x0AF7
       mov      rdi, 0x76DAAF74AEA0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG62:                ;; offset=0x0B0B
       mov      rdi, 0x76DAAF74AEA4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG63:                ;; offset=0x0B1F
       mov      rdi, 0x76DAAF74AEA8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG64:                ;; offset=0x0B33
       mov      rdi, 0x76DAAF74AEAC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG65:                ;; offset=0x0B47
       mov      rdi, 0x76DAAF74AEB0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG66:                ;; offset=0x0B5B
       mov      rdi, 0x76DAAF74AEB4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG67:                ;; offset=0x0B6F
       mov      rdi, 0x76DAAF74AEB8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG68:                ;; offset=0x0B83
       mov      rdi, 0x76DAAF74AEBC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG69:                ;; offset=0x0B97
       mov      rdi, 0x76DAAF74AEC0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG70:                ;; offset=0x0BAB
       mov      rdi, 0x76DAAF74AEC4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG71:                ;; offset=0x0BBF
       mov      rdi, 0x76DAAF74AEC8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG72:                ;; offset=0x0BD3
       mov      rdi, 0x76DAAF74AECC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG73:                ;; offset=0x0BE7
       mov      rdi, 0x76DAAF74AED0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG74:                ;; offset=0x0BFB
       mov      rdi, 0x76DAAF74AED4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG75:                ;; offset=0x0C0F
       mov      rdi, 0x76DAAF74AED8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG76:                ;; offset=0x0C23
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG77
       mov      rdi, 0x76DAAF74AEDC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG78
 
G_M000_IG77:                ;; offset=0x0C3D
       cmp      dword ptr [rbp+0x80], 8
       jne      SHORT G_M000_IG79
       mov      rdi, 0x76DAAF74AEE0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG79
 
G_M000_IG78:                ;; offset=0x0C57
       mov      rdi, 0x76DAAF34A470
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0x418], rax
       mov      rdi, gword ptr [rbp-0x418]
       call     [System.PlatformNotSupportedException:.ctor():this]
       mov      rdi, gword ptr [rbp-0x418]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG79:                ;; offset=0x0C87
       mov      rdi, bword ptr [rbp-0x38]
       mov      rsi, qword ptr [rbp-0x30]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG81
       mov      rdi, bword ptr [rbp-0x48]
       mov      rsi, qword ptr [rbp-0x40]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG83
       mov      rdi, bword ptr [rbp-0x58]
       mov      rsi, qword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG82
       mov      rdi, bword ptr [rbp+0x10]
       mov      rsi, qword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG84
 
G_M000_IG80:                ;; offset=0x0CCF
       mov      rdi, 0x76DAAF74AEE4
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG81:                ;; offset=0x0CDE
       mov      rdi, 0x76DAAF74AEE8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG97
 
G_M000_IG82:                ;; offset=0x0CF2
       mov      rdi, 0x76DAAF74AEEC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG81
 
G_M000_IG83:                ;; offset=0x0D03
       mov      rdi, 0x76DAAF74AEF0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG81
 
G_M000_IG84:                ;; offset=0x0D14
       mov      eax, dword ptr [rbp+0x78]
       inc      eax
       mov      dword ptr [rbp-0x454], eax
       mov      eax, dword ptr [rbp-0x454]
       shr      eax, 31
       add      eax, dword ptr [rbp-0x454]
       sar      eax, 1
       mov      dword ptr [rbp-0x74], eax
       mov      eax, dword ptr [rbp+0x70]
       add      eax, 1
       jo       G_M000_IG103
       mov      dword ptr [rbp-0x458], eax
       mov      eax, dword ptr [rbp-0x458]
       shr      eax, 31
       add      eax, dword ptr [rbp-0x458]
       sar      eax, 1
       imul     eax, dword ptr [rbp-0x74]
       jo       G_M000_IG103
       mov      dword ptr [rbp-0x78], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x7C], eax
       jmp      G_M000_IG92
 
G_M000_IG85:                ;; offset=0x0D6D
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
       jne      SHORT G_M000_IG86
       mov      rdi, 0x76DAAF74AEF4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG97
 
G_M000_IG86:                ;; offset=0x0E04
       lea      rdi, [rbp+0x30]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xA8], rax
       mov      rax, bword ptr [rbp-0xA8]
       mov      qword ptr [rbp-0x438], rax
       mov      rax, qword ptr [rbp-0x438]
       mov      qword ptr [rbp-0x88], rax
       lea      rdi, [rbp-0x48]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xB0], rax
       mov      rax, bword ptr [rbp-0xB0]
       mov      qword ptr [rbp-0x440], rax
       mov      rax, qword ptr [rbp-0x440]
       mov      qword ptr [rbp-0x90], rax
       lea      rdi, [rbp+0x40]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xB8], rax
       mov      rax, bword ptr [rbp-0xB8]
       mov      qword ptr [rbp-0x448], rax
       mov      rax, qword ptr [rbp-0x448]
       mov      qword ptr [rbp-0x98], rax
       lea      rdi, [rbp+0x50]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xC0], rax
       mov      rax, bword ptr [rbp-0xC0]
       mov      qword ptr [rbp-0x450], rax
       mov      rax, qword ptr [rbp-0x450]
       mov      qword ptr [rbp-0xA0], rax
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG87
       mov      rdi, qword ptr [rbp-0x88]
       mov      rsi, qword ptr [rbp-0x90]
       mov      rdx, qword ptr [rbp-0x98]
       mov      ecx, dword ptr [rbp+0x60]
       mov      r8d, dword ptr [rbp+0x68]
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int)]
       jmp      SHORT G_M000_IG88
 
G_M000_IG87:                ;; offset=0x0EE5
       mov      rdi, 0x76DAAF74AEF8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rdi, qword ptr [rbp-0x88]
       mov      rsi, qword ptr [rbp-0x90]
       mov      rdx, qword ptr [rbp-0x98]
       mov      ecx, dword ptr [rbp+0x60]
       mov      r8d, dword ptr [rbp+0x68]
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int)]
 
G_M000_IG88:                ;; offset=0x0F16
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3E0], rax
       mov      qword ptr [rbp-0x3D8], rdx
       mov      rdi, bword ptr [rbp-0x3E0]
       mov      rsi, qword ptr [rbp-0x3D8]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG89
       mov      rdi, 0x76DAAF74AEFC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG97
 
G_M000_IG89:                ;; offset=0x0F5E
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG90
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
       jmp      SHORT G_M000_IG91
 
G_M000_IG90:                ;; offset=0x0F98
       mov      rdi, 0x76DAAF74AF00
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
 
G_M000_IG91:                ;; offset=0x0FD6
       mov      rdi, 0x76DAAF74AF04
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
 
G_M000_IG92:                ;; offset=0x1012
       mov      eax, dword ptr [rbp-0x420]
       dec      eax
       mov      dword ptr [rbp-0x420], eax
       cmp      dword ptr [rbp-0x420], 0
       jg       SHORT G_M000_IG94
 
G_M000_IG93:                ;; offset=0x1029
       lea      rdi, [rbp-0x420]
       mov      esi, 943
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG94:                ;; offset=0x103A
       mov      eax, dword ptr [rbp-0x7C]
       cmp      eax, dword ptr [rbp-0x78]
       jl       G_M000_IG85
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3F0], rax
       mov      qword ptr [rbp-0x3E8], rdx
       mov      rdi, bword ptr [rbp-0x3F0]
       mov      rsi, qword ptr [rbp-0x3E8]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG96
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x400], rax
       mov      qword ptr [rbp-0x3F8], rdx
       mov      rdi, bword ptr [rbp-0x400]
       mov      rsi, qword ptr [rbp-0x3F8]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG100
       mov      rdi, bword ptr [rbp-0x58]
       mov      rsi, qword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG99
       mov      rdi, bword ptr [rbp+0x10]
       mov      rsi, qword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG101
 
G_M000_IG95:                ;; offset=0x10D2
       mov      rdi, 0x76DAAF74AF08
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG96:                ;; offset=0x10E1
       mov      rdi, 0x76DAAF74AF0C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG97:                ;; offset=0x10F0
       xor      eax, eax
 
G_M000_IG98:                ;; offset=0x10F2
       add      rsp, 0x490
       pop      rbp
       ret      
 
G_M000_IG99:                ;; offset=0x10FB
       mov      rdi, 0x76DAAF74AF10
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG96
 
G_M000_IG100:                ;; offset=0x110C
       mov      rdi, 0x76DAAF74AF14
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG96
 
G_M000_IG101:                ;; offset=0x111D
       mov      rdi, 0x76DAAF74AF18
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
 
G_M000_IG102:                ;; offset=0x11B1
       add      rsp, 0x490
       pop      rbp
       ret      
 
G_M000_IG103:                ;; offset=0x11BA
       call     CORINFO_HELP_OVERFLOW
       int3     
 
; Total bytes of code 4544

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int) (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       sub      rsp, 136
       lea      rbp, [rsp+0xB0]
       xor      eax, eax
       mov      qword ptr [rbp-0x48], rax
       mov      qword ptr [rbp-0x50], rax
       mov      qword ptr [rbp-0x88], 0x1EA9B682
       mov      r11, rdi
       mov      r13d, r8d
       mov      r8d, esi
       mov      r14d, r9d
       mov      r9, rdx
       mov      r15d, dword ptr [rbp+0x10]
       mov      r12d, dword ptr [rbp+0x18]
       mov      r10d, dword ptr [rbp+0x20]
       mov      ebx, dword ptr [rbp+0x28]
 
G_M000_IG02:                ;; offset=0x004C
       mov      eax, r10d
       cdq      
       idiv     edx:eax, r12d
       mov      edi, eax
       lea      esi, [2*rdi-0x01]
       mov      eax, r10d
       cdq      
       idiv     edx:eax, r12d
       lea      edx, [2*rdx-0x01]
       mov      dword ptr [rbp-0x30], edx
       cmp      ebx, 8
       jne      G_M000_IG13
 
G_M000_IG03:                ;; offset=0x0076
       mov      dword ptr [rbp+0x20], r10d
       lea      eax, [r10+0x07]
       cdq      
       idiv     edx:eax, r12d
       cmp      eax, edi
       jne      SHORT G_M000_IG12
 
G_M000_IG04:                ;; offset=0x0086
       test     esi, esi
       jl       SHORT G_M000_IG11
 
G_M000_IG05:                ;; offset=0x008A
       lea      edi, [rsi+0x03]
       cmp      edi, r14d
       jge      SHORT G_M000_IG10
 
G_M000_IG06:                ;; offset=0x0092
       mov      edi, dword ptr [rbp-0x30]
       test     edi, edi
       jl       SHORT G_M000_IG09
 
G_M000_IG07:                ;; offset=0x0099
       lea      edx, [rdi+0x11]
       cmp      edx, r15d
       jl       G_M000_IG41
 
G_M000_IG08:                ;; offset=0x00A5
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG14
       align    [0 bytes for IG19]
 
G_M000_IG09:                ;; offset=0x00B6
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG14
 
G_M000_IG10:                ;; offset=0x00C7
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG14
 
G_M000_IG11:                ;; offset=0x00D8
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG14
 
G_M000_IG12:                ;; offset=0x00E9
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
       jmp      SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x00FA
       mov      dword ptr [rbp+0x20], r10d
       mov      bword ptr [rbp-0x78], r11
       mov      bword ptr [rbp-0x80], r9
       mov      dword ptr [rbp-0x54], r8d
       mov      dword ptr [rbp-0x58], ecx
 
G_M000_IG14:                ;; offset=0x010D
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp+0x20]
       mov      qword ptr [rbp-0x60], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x60]
       mov      qword ptr [rbp-0x38], rdi
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp+0x20]
       mov      qword ptr [rbp-0x68], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x68]
       mov      qword ptr [rbp-0x40], rdi
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp+0x20]
       mov      qword ptr [rbp-0x70], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x70]
       mov      rcx, rdi
       xor      edi, edi
       jmp      SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x0182
       inc      edi
       cmp      edi, 8
       mov      r12d, dword ptr [rbp+0x18]
       jge      G_M000_IG28
 
G_M000_IG16:                ;; offset=0x0191
       mov      esi, edi
       add      esi, dword ptr [rbp+0x20]
       mov      eax, esi
       cdq      
       idiv     edx:eax, r12d
       lea      r8d, [2*rax-0x01]
       mov      dword ptr [rbp+0x18], r12d
       mov      eax, esi
       cdq      
       idiv     edx:eax, r12d
       lea      eax, [2*rdx-0x01]
       xor      edx, edx
       jmp      SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x01B9
       inc      edx
       cmp      edx, 4
       jge      SHORT G_M000_IG15
 
G_M000_IG18:                ;; offset=0x01C0
       xor      esi, esi
       jmp      SHORT G_M000_IG25
 
G_M000_IG19:                ;; offset=0x01C4
       xor      r11d, r11d
 
G_M000_IG20:                ;; offset=0x01C7
       movsxd   r9, r9d
       shl      r9, 2
       mov      r10, qword ptr [rbp-0x38]
       lea      r12, [r10+r9]
       test     r11d, r11d
       jne      G_M000_IG39
 
G_M000_IG21:                ;; offset=0x01DF
       mov      dword ptr [rbp-0x2C], r13d
       xor      r13d, r13d
 
G_M000_IG22:                ;; offset=0x01E6
       mov      dword ptr [r12], r13d
       mov      r13, qword ptr [rbp-0x40]
       add      r9, r13
       test     r11d, r11d
       jne      G_M000_IG40
 
G_M000_IG23:                ;; offset=0x01FA
       xor      r11d, r11d
 
G_M000_IG24:                ;; offset=0x01FD
       mov      dword ptr [r9], r11d
       inc      esi
       cmp      esi, 4
       mov      r13d, dword ptr [rbp-0x2C]
       jge      SHORT G_M000_IG17
 
G_M000_IG25:                ;; offset=0x020B
       lea      r9d, [rsi+4*rdx]
       lea      r9d, [rdi+8*r9]
       cmp      edi, ebx
       jge      SHORT G_M000_IG19
 
G_M000_IG26:                ;; offset=0x0217
       lea      r11d, [r8+rdx]
       cmp      r11d, r14d
       jae      SHORT G_M000_IG19
 
G_M000_IG27:                ;; offset=0x0220
       lea      r11d, [rax+rsi]
       cmp      r11d, r15d
       setb     r11b
       movzx    r11, r11b
       jmp      SHORT G_M000_IG20
 
G_M000_IG28:                ;; offset=0x0231
       xor      rax, rax
       cmp      dword ptr [rbp-0x54], 0
       cmovne   rax, bword ptr [rbp-0x78]
       mov      bword ptr [rbp-0x48], rax
       xor      rdx, rdx
       cmp      dword ptr [rbp-0x58], 0
       cmovne   rdx, bword ptr [rbp-0x80]
       mov      bword ptr [rbp-0x50], rdx
       xor      edi, edi
       cmp      edi, r13d
       jge      G_M000_IG34
 
G_M000_IG29:                ;; offset=0x025A
       mov      esi, edi
       imul     esi, r14d
       imul     esi, r15d
       movsxd   rsi, esi
       lea      rsi, [rax+4*rsi]
       xor      r8d, r8d
       mov      r9d, 4
       align    [0 bytes for IG30]
 
G_M000_IG30:                ;; offset=0x0274
       lea      r10d, [8*r8]
       vxorps   ymm0, ymm0, ymm0
       movsxd   r10, r10d
       mov      rbx, qword ptr [rbp-0x38]
       vmovups  ymm1, ymmword ptr [rbx+4*r10]
       mov      r12, qword ptr [rbp-0x40]
       vmovups  ymm2, ymmword ptr [r12+4*r10]
       vmovaps  ymm3, ymm2
       vgatherdps ymm0, dword ptr [rsi+4*xmm1], ymm3
       vxorps   ymm1, ymm1, ymm1
       vmovups  ymm2, ymmword ptr [rbx+4*r10+0x20]
       vmovups  ymm3, ymmword ptr [r12+4*r10+0x20]
       vmovaps  ymm4, ymm3
       vgatherdps ymm1, dword ptr [rsi+4*xmm2], ymm4
       vxorps   ymm2, ymm2, ymm2
       vmovups  ymm3, ymmword ptr [rbx+4*r10+0x40]
       vmovups  ymm4, ymmword ptr [r12+4*r10+0x40]
       vmovaps  ymm5, ymm4
       vgatherdps ymm2, dword ptr [rsi+4*xmm3], ymm5
       vxorps   ymm3, ymm3, ymm3
       vmovups  ymm4, ymmword ptr [rbx+4*r10+0x60]
       vmovups  ymm5, ymmword ptr [r12+4*r10+0x60]
       vmovaps  ymm6, ymm5
       vgatherdps ymm3, dword ptr [rsi+4*xmm4], ymm6
       vsubps   ymm0, ymm0, ymm2
       movsxd   r10, r8d
       shl      r10, 5
       vmovups  ymmword ptr [rcx+r10], ymm0
       vaddps   ymm0, ymm1, ymm2
       lea      r10d, [r8+0x01]
       movsxd   r10, r10d
       shl      r10, 5
       vmovups  ymmword ptr [rcx+r10], ymm0
       vsubps   ymm0, ymm2, ymm1
       lea      r10d, [r8+0x02]
       movsxd   r10, r10d
       shl      r10, 5
       vmovups  ymmword ptr [rcx+r10], ymm0
       vsubps   ymm0, ymm1, ymm3
       lea      r10d, [r8+0x03]
       movsxd   r10, r10d
       shl      r10, 5
       vmovups  ymmword ptr [rcx+r10], ymm0
       add      r8d, 4
       dec      r9d
       jne      G_M000_IG30
 
G_M000_IG31:                ;; offset=0x0352
       xor      esi, esi
       align    [0 bytes for IG32]
 
G_M000_IG32:                ;; offset=0x0354
       movsxd   r8, esi
       shl      r8, 5
       vmovups  ymm0, ymmword ptr [rcx+r8]
       lea      r8d, [rsi+0x04]
       movsxd   r9, r8d
       shl      r9, 5
       vmovups  ymm1, ymmword ptr [rcx+r9]
       lea      r9d, [rsi+0x08]
       movsxd   r10, r9d
       shl      r10, 5
       vmovups  ymm2, ymmword ptr [rcx+r10]
       lea      r10d, [rsi+0x0C]
       movsxd   r11, r10d
       shl      r11, 5
       vmovups  ymm3, ymmword ptr [rcx+r11]
       vsubps   ymm0, ymm0, ymm2
       mov      r11d, esi
       imul     r11d, r13d
       add      r11d, edi
       shl      r11d, 3
       movsxd   r11, r11d
       vmovups  ymmword ptr [rdx+4*r11], ymm0
       vaddps   ymm0, ymm1, ymm2
       imul     r8d, r13d
       add      r8d, edi
       shl      r8d, 3
       movsxd   r8, r8d
       vmovups  ymmword ptr [rdx+4*r8], ymm0
       vsubps   ymm0, ymm2, ymm1
       imul     r9d, r13d
       add      r9d, edi
       shl      r9d, 3
       movsxd   r8, r9d
       vmovups  ymmword ptr [rdx+4*r8], ymm0
       vsubps   ymm0, ymm1, ymm3
       imul     r10d, r13d
       add      r10d, edi
       shl      r10d, 3
       movsxd   r8, r10d
       vmovups  ymmword ptr [rdx+4*r8], ymm0
       inc      esi
       cmp      esi, 4
       jl       G_M000_IG32
 
G_M000_IG33:                ;; offset=0x0402
       inc      edi
       cmp      edi, r13d
       jl       G_M000_IG29
 
G_M000_IG34:                ;; offset=0x040D
       xor      eax, eax
       mov      bword ptr [rbp-0x48], rax
 
G_M000_IG35:                ;; offset=0x0413
       mov      bword ptr [rbp-0x50], rax
 
G_M000_IG36:                ;; offset=0x0417
       cmp      qword ptr [rbp-0x88], 0x1EA9B682
       je       SHORT G_M000_IG37
       call     CORINFO_HELP_FAIL_FAST
 
G_M000_IG37:                ;; offset=0x0429
       nop      
 
G_M000_IG38:                ;; offset=0x042A
       vzeroupper 
       lea      rsp, [rbp-0x28]
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG39:                ;; offset=0x043C
       mov      dword ptr [rbp-0x2C], r13d
       lea      r13d, [r8+rdx]
       imul     r13d, r15d
       add      r13d, eax
       add      r13d, esi
       jmp      G_M000_IG22
 
G_M000_IG40:                ;; offset=0x0453
       mov      r11d, -1
       jmp      G_M000_IG24
 
G_M000_IG41:                ;; offset=0x045E
       mov      dword ptr [rsp], r15d
       mov      dword ptr [rsp+0x08], esi
       mov      dword ptr [rsp+0x10], edi
       mov      rdi, r11
       mov      esi, r8d
       mov      rdx, r9
       mov      r8d, r13d
       mov      r9d, r14d
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int)]
       jmp      SHORT G_M000_IG36
 
; Total bytes of code 1153

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 416
       lea      rbp, [rsp+0x1A0]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x190], xmm8
       vmovdqa  xmmword ptr [rbp-0x180], xmm8
       mov      rax, -288
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
 
G_M000_IG02:                ;; offset=0x005E
       mov      dword ptr [rbp-0x1A0], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x4C], eax
       jmp      G_M000_IG12
 
G_M000_IG03:                ;; offset=0x0072
       xor      eax, eax
       mov      dword ptr [rbp-0x50], eax
       jmp      G_M000_IG09
 
G_M000_IG04:                ;; offset=0x007C
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x70], ymm0
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
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       imul     eax, dword ptr [rbp-0x48]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       movsxd   rcx, dword ptr [rbp-0x50]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x158], rax
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x160], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x164], eax
       jmp      G_M000_IG06
 
G_M000_IG05:                ;; offset=0x0125
       mov      rdi, 0x76DAAF74AFC8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x158]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x190], ymm0
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax]
       vmovups  ymm1, ymmword ptr [rbp-0x70]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x70], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x04]
       vmovups  ymm1, ymmword ptr [rbp-0x90]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x90], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x08]
       vmovups  ymm1, ymmword ptr [rbp-0xB0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xB0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x0C]
       vmovups  ymm1, ymmword ptr [rbp-0xD0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xD0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x10]
       vmovups  ymm1, ymmword ptr [rbp-0xF0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xF0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x14]
       vmovups  ymm1, ymmword ptr [rbp-0x110]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x110], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x18]
       vmovups  ymm1, ymmword ptr [rbp-0x130]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x130], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x1C]
       vmovups  ymm1, ymmword ptr [rbp-0x150]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x150], ymm1
       movsxd   rax, dword ptr [rbp-0x48]
       mov      rcx, qword ptr [rbp-0x158]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x158], rax
       mov      rax, qword ptr [rbp-0x160]
       add      rax, 32
       mov      qword ptr [rbp-0x160], rax
       mov      eax, dword ptr [rbp-0x164]
       inc      eax
       mov      dword ptr [rbp-0x164], eax
 
G_M000_IG06:                ;; offset=0x02A6
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x02BD
       lea      rdi, [rbp-0x1A0]
       mov      esi, 320
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x02CE
       mov      eax, dword ptr [rbp-0x164]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG05
       mov      rdi, 0x76DAAF74AFCC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x48]
       add      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x198], rax
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vmovups  ymmword ptr [rax], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x90]
       vmovups  ymmword ptr [rax+0x20], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xB0]
       vmovups  ymmword ptr [rax+0x40], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xD0]
       vmovups  ymmword ptr [rax+0x60], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xF0]
       vmovups  ymmword ptr [rax+0x80], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x110]
       vmovups  ymmword ptr [rax+0xA0], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x130]
       vmovups  ymmword ptr [rax+0xC0], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x150]
       vmovups  ymmword ptr [rax+0xE0], ymm0
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 8
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG09:                ;; offset=0x03BB
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x03D2
       lea      rdi, [rbp-0x1A0]
       mov      esi, 447
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG11:                ;; offset=0x03E3
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp-0x48]
       jl       G_M000_IG04
       mov      rdi, 0x76DAAF74AFD0
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       inc      eax
       mov      dword ptr [rbp-0x4C], eax
 
G_M000_IG12:                ;; offset=0x0406
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x041D
       lea      rdi, [rbp-0x1A0]
       mov      esi, 459
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG14:                ;; offset=0x042E
       cmp      dword ptr [rbp-0x4C], 16
       jl       G_M000_IG03
       mov      rdi, 0x76DAAF74AFD4
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG15:                ;; offset=0x0448
       vzeroupper 
       add      rsp, 416
       pop      rbp
       ret      
 
; Total bytes of code 1108

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
       mov      rdi, 0x76DAAF74B018
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
       mov      rdi, 0x76DAAF74B01C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
       lea      eax, [8*rax+0x08]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1D8]
       vmovups  ymm0, ymmword ptr [rbp-0x1D0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG09:                ;; offset=0x03FC
       mov      rdi, 0x76DAAF74B020
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
       mov      rdi, 0x76DAAF74B024
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
       mov      rdi, 0x76DAAF74B028
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG16:                ;; offset=0x04A3
       vzeroupper 
       add      rsp, 576
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x04AF
       mov      rdi, 0x76DAAF74B02C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG09
 
; Total bytes of code 1219

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x140
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 5

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 16
       mov      qword ptr [rsp+0x1B8], r15
       mov      qword ptr [rsp+0x1B0], rbx
       lea      rbp, [rsp+0x10]
       mov      rdi, qword ptr [rbp+0x180]
       mov      rsi, qword ptr [rbp+0x178]
       mov      rdx, qword ptr [rbp+0x170]
       mov      ecx, dword ptr [rbp+0x16C]
       mov      eax, dword ptr [rbp+0x168]
       mov      ebx, dword ptr [rbp+0x164]
       mov      r11d, dword ptr [rbp+0x160]
       vmovups  ymm0, ymmword ptr [rbp+0x140]
       vmovups  ymm1, ymmword ptr [rbp+0x120]
       vmovups  ymm2, ymmword ptr [rbp+0x100]
       vmovups  ymm3, ymmword ptr [rbp+0xE0]
       vmovups  ymm4, ymmword ptr [rbp+0xC0]
       vmovups  ymm5, ymmword ptr [rbp+0xA0]
       vmovups  ymm6, ymmword ptr [rbp+0x80]
       vmovups  ymm7, ymmword ptr [rbp+0x60]
       mov      r10, qword ptr [rbp+0x58]
       mov      r8, qword ptr [rbp+0x50]
       mov      r9d, dword ptr [rbp+0x4C]
 
G_M000_IG02:                ;; offset=0x0095
       jmp      G_M000_IG05
       align    [8 bytes for IG07]
 
G_M000_IG03:                ;; offset=0x00A2
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r8d, ebx
       imul     r8d, ecx
       mov      r9d, r8d
       imul     r9d, eax
       movsxd   r9, r9d
       shl      r9, 2
       add      r9, rsi
       movsxd   r10, r11d
       lea      r10, [r9+4*r10]
       shl      r8d, 3
       movsxd   r8, r8d
       lea      r8, [rdi+4*r8]
       xor      r9d, r9d
       cmp      r9d, ecx
       jge      SHORT G_M000_IG06
 
G_M000_IG04:                ;; offset=0x00F4
       vmovups  ymm8, ymmword ptr [r10]
       vbroadcastss ymm9, dword ptr [r8]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       movsxd   r15, eax
       lea      r10, [r10+4*r15]
       add      r8, 32
       inc      r9d
 
G_M000_IG05:                ;; offset=0x015E
       cmp      r9d, ecx
       jl       SHORT G_M000_IG04
 
G_M000_IG06:                ;; offset=0x0163
       mov      r10d, ebx
       imul     r10d, eax
       add      r10d, r11d
       shl      r10d, 3
       movsxd   r8, r10d
       lea      r9, [rdx+4*r8]
       vmovups  ymmword ptr [r9], ymm0
       vmovups  ymmword ptr [r9+0x20], ymm1
       vmovups  ymmword ptr [r9+0x40], ymm2
       vmovups  ymmword ptr [r9+0x60], ymm3
       vmovups  ymmword ptr [r9+0x80], ymm4
       vmovups  ymmword ptr [r9+0xA0], ymm5
       vmovups  ymmword ptr [r9+0xC0], ymm6
       vmovups  ymmword ptr [r9+0xE0], ymm7
       add      r11d, 8
       cmp      r11d, eax
       jl       G_M000_IG03
 
G_M000_IG07:                ;; offset=0x01C0
       inc      ebx
       cmp      ebx, 16
       jge      SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x01C7
       xor      r11d, r11d
       test     eax, eax
       jg       G_M000_IG03
       jmp      SHORT G_M000_IG07
 
G_M000_IG09:                ;; offset=0x01D4
       vzeroupper 
       add      rsp, 432
       pop      rbx
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 483

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int) (Tier0-FullOpts)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       sub      rsp, 104
       lea      rbp, [rsp+0x90]
       xor      eax, eax
       mov      qword ptr [rbp-0x58], rax
       mov      qword ptr [rbp-0x60], rax
       mov      qword ptr [rbp-0x88], 0x1EA9B682
       mov      bword ptr [rbp-0x78], rdi
       mov      dword ptr [rbp-0x64], esi
       mov      bword ptr [rbp-0x80], rdx
       mov      dword ptr [rbp-0x68], ecx
       mov      ebx, r8d
       mov      r13d, r9d
       mov      r15d, dword ptr [rbp+0x10]
       mov      r14d, dword ptr [rbp+0x18]
       mov      r12d, dword ptr [rbp+0x20]
 
G_M000_IG02:                ;; offset=0x004B
       vmovups  ymm0, ymmword ptr [reloc @RWD00]
       vmovups  ymmword ptr [rbp-0x50], ymm0
       test     dword ptr [rsp], esp
       sub      rsp, 512
       lea      rdi, [rsp]
       mov      qword ptr [rbp-0x70], rdi
       mov      esi, 512
       call     [CORINFO_HELP_MEMZERO]
       mov      rdi, qword ptr [rbp-0x70]
       mov      rax, rdi
       xor      rcx, rcx
       cmp      dword ptr [rbp-0x64], 0
       cmovne   rcx, bword ptr [rbp-0x78]
       mov      bword ptr [rbp-0x58], rcx
       xor      rdx, rdx
       cmp      dword ptr [rbp-0x68], 0
       cmovne   rdx, bword ptr [rbp-0x80]
       mov      bword ptr [rbp-0x60], rdx
       xor      edi, edi
       cmp      edi, ebx
       jge      G_M000_IG08
 
G_M000_IG03:                ;; offset=0x00A4
       mov      esi, edi
       imul     esi, r13d
       imul     esi, r15d
       movsxd   rsi, esi
       lea      rsi, [rcx+4*rsi]
       xor      r8d, r8d
       movsxd   r9, r12d
       shl      r9, 2
       align    [0 bytes for IG04]
 
G_M000_IG04:                ;; offset=0x00BF
       lea      r10d, [r14+r8]
       imul     r10d, r15d
       movsxd   r10, r10d
       shl      r10, 2
       add      r10, rsi
       add      r10, r9
       vmovups  ymm0, ymmword ptr [rbp-0x50]
       vpermps  ymm1, ymm0, ymmword ptr [r10]
       vpermps  ymm2, ymm0, ymmword ptr [r10+0x20]
       vpermps  ymm3, ymm0, ymmword ptr [r10+0x08]
       vpermps  ymm4, ymm0, ymmword ptr [r10+0x28]
       vperm2f128 ymm5, ymm1, ymm2, 32
       vperm2f128 ymm1, ymm1, ymm2, 49
       vperm2f128 ymm2, ymm3, ymm4, 32
       vperm2f128 ymm3, ymm3, ymm4, 49
       vsubps   ymm4, ymm5, ymm2
       lea      r10d, [4*r8]
       movsxd   r11, r10d
       shl      r11, 5
       vmovups  ymmword ptr [rax+r11], ymm4
       vaddps   ymm4, ymm1, ymm2
       lea      r11d, [r10+0x01]
       movsxd   r11, r11d
       shl      r11, 5
       vmovups  ymmword ptr [rax+r11], ymm4
       vsubps   ymm2, ymm2, ymm1
       lea      r11d, [r10+0x02]
       movsxd   r11, r11d
       shl      r11, 5
       vmovups  ymmword ptr [rax+r11], ymm2
       vsubps   ymm1, ymm1, ymm3
       add      r10d, 3
       movsxd   r10, r10d
       shl      r10, 5
       vmovups  ymmword ptr [rax+r10], ymm1
       inc      r8d
       cmp      r8d, 4
       jl       G_M000_IG04
 
G_M000_IG05:                ;; offset=0x016D
       xor      esi, esi
       align    [0 bytes for IG06]
 
G_M000_IG06:                ;; offset=0x016F
       movsxd   r8, esi
       shl      r8, 5
       vmovups  ymm1, ymmword ptr [rax+r8]
       lea      r8d, [rsi+0x04]
       movsxd   r9, r8d
       shl      r9, 5
       vmovups  ymm2, ymmword ptr [rax+r9]
       lea      r9d, [rsi+0x08]
       movsxd   r10, r9d
       shl      r10, 5
       vmovups  ymm3, ymmword ptr [rax+r10]
       lea      r10d, [rsi+0x0C]
       movsxd   r11, r10d
       shl      r11, 5
       vmovups  ymm4, ymmword ptr [rax+r11]
       vsubps   ymm1, ymm1, ymm3
       mov      r11d, esi
       imul     r11d, ebx
       add      r11d, edi
       shl      r11d, 3
       movsxd   r11, r11d
       vmovups  ymmword ptr [rdx+4*r11], ymm1
       vaddps   ymm1, ymm2, ymm3
       imul     r8d, ebx
       add      r8d, edi
       shl      r8d, 3
       movsxd   r8, r8d
       vmovups  ymmword ptr [rdx+4*r8], ymm1
       vsubps   ymm1, ymm3, ymm2
       imul     r9d, ebx
       add      r9d, edi
       shl      r9d, 3
       movsxd   r8, r9d
       vmovups  ymmword ptr [rdx+4*r8], ymm1
       vsubps   ymm1, ymm2, ymm4
       imul     r10d, ebx
       add      r10d, edi
       shl      r10d, 3
       movsxd   r8, r10d
       vmovups  ymmword ptr [rdx+4*r8], ymm1
       inc      esi
       cmp      esi, 4
       jl       G_M000_IG06
 
G_M000_IG07:                ;; offset=0x021D
       inc      edi
       cmp      edi, ebx
       jl       G_M000_IG03
 
G_M000_IG08:                ;; offset=0x0227
       xor      eax, eax
       mov      bword ptr [rbp-0x58], rax
 
G_M000_IG09:                ;; offset=0x022D
       mov      bword ptr [rbp-0x60], rax
       cmp      qword ptr [rbp-0x88], 0x1EA9B682
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
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 288
       lea      rbp, [rsp+0x120]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x110], xmm8
       mov      rax, -192
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x20], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x40], rax
       mov      bword ptr [rbp-0x38], rdi
       mov      qword ptr [rbp-0x30], rsi
 
G_M000_IG02:                ;; offset=0x004B
       mov      dword ptr [rbp-0x118], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x3C], eax
       lea      rdi, [rbp-0x38]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0x50], rax
       mov      rax, bword ptr [rbp-0x50]
       mov      qword ptr [rbp-0x120], rax
       mov      rax, qword ptr [rbp-0x120]
       mov      qword ptr [rbp-0x48], rax
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       vmovups  ymmword ptr [rbp-0xF0], ymm0
       vbroadcastss ymm0, dword ptr [reloc @RWD04]
       vmovups  ymmword ptr [rbp-0x110], ymm0
       jmp      SHORT G_M000_IG06
 
G_M000_IG03:                ;; offset=0x00A2
       movsxd   rax, dword ptr [rbp-0x3C]
       mov      rcx, qword ptr [rbp-0x48]
       vmovups  ymm0, ymmword ptr [rcx+4*rax]
       vpand    ymm0, ymm0, ymmword ptr [rbp-0xF0]
       vcmpgtps ymm0, ymm0, ymmword ptr [rbp-0x110]
       vmovmskps rax, ymm0
       test     eax, eax
       je       SHORT G_M000_IG05
       mov      rdi, 0x76DAAF750FC8
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG04:                ;; offset=0x00D9
       vzeroupper 
       add      rsp, 288
       pop      rbp
       ret      
 
G_M000_IG05:                ;; offset=0x00E5
       mov      rdi, 0x76DAAF750FCC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       add      eax, 8
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG06:                ;; offset=0x00FD
       mov      eax, dword ptr [rbp-0x118]
       dec      eax
       mov      dword ptr [rbp-0x118], eax
       cmp      dword ptr [rbp-0x118], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x0114
       lea      rdi, [rbp-0x118]
       mov      esi, 191
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x0125
       mov      eax, dword ptr [rbp-0x30]
       add      eax, -8
       cmp      dword ptr [rbp-0x3C], eax
       jle      G_M000_IG03
 
G_M000_IG09:                ;; offset=0x0134
       mov      rdi, 0x76DAAF750FD0
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
       mov      bword ptr [rbp-0x50], rax
       jmp      SHORT G_M000_IG13
 
G_M000_IG10:                ;; offset=0x014B
       mov      eax, dword ptr [rbp-0x30]
       cmp      dword ptr [rbp-0x3C], eax
       jae      G_M000_IG17
       mov      eax, dword ptr [rbp-0x3C]
       mov      rcx, bword ptr [rbp-0x38]
       vmovss   xmm0, dword ptr [rcx+4*rax]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       jbe      SHORT G_M000_IG12
       mov      rdi, 0x76DAAF750FD4
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG11:                ;; offset=0x0186
       vzeroupper 
       add      rsp, 288
       pop      rbp
       ret      
 
G_M000_IG12:                ;; offset=0x0192
       mov      rdi, 0x76DAAF750FD8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       inc      eax
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG13:                ;; offset=0x01A9
       mov      eax, dword ptr [rbp-0x118]
       dec      eax
       mov      dword ptr [rbp-0x118], eax
       cmp      dword ptr [rbp-0x118], 0
       jg       SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x01C0
       lea      rdi, [rbp-0x118]
       mov      esi, 235
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG15:                ;; offset=0x01D1
       mov      eax, dword ptr [rbp-0x3C]
       cmp      eax, dword ptr [rbp-0x30]
       jl       G_M000_IG10
       mov      rdi, 0x76DAAF750FDC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, 1
 
G_M000_IG16:                ;; offset=0x01F1
       vzeroupper 
       add      rsp, 288
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x01FD
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	7E7FFFFFh		; 8.50706e+37
RWD08  	dd	00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 515

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0xbf
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
       mov      eax, dword ptr [rbp+0xF4]
       mov      rcx, qword ptr [rbp+0xE8]
       vmovups  ymm0, ymmword ptr [rbp+0x40]
       vmovups  ymm1, ymmword ptr [rbp+0x20]
 
G_M000_IG02:                ;; offset=0x001F
       mov      rdx, bword ptr [rbp+0xF8]
       mov      edi, dword ptr [rbp+0x100]
       lea      esi, [rdi-0x08]
       cmp      eax, esi
       jg       SHORT G_M000_IG05
       align    [13 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0040
       movsxd   r8, eax
       vpand    ymm2, ymm0, ymmword ptr [rcx+4*r8]
       vcmpgtps ymm2, ymm2, ymm1
       vmovmskps r8, ymm2
       test     r8d, r8d
       jne      SHORT G_M000_IG12
 
G_M000_IG04:                ;; offset=0x0057
       add      eax, 8
       cmp      eax, esi
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x005E
       xor      ecx, ecx
       mov      bword ptr [rbp+0xE0], rcx
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
       add      rsp, 304
       pop      rbp
       ret      
 
G_M000_IG12:                ;; offset=0x00A3
       xor      eax, eax
 
G_M000_IG13:                ;; offset=0x00A5
       vzeroupper 
       add      rsp, 304
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
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x3af
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 5
; 8 inlinees with PGO data; 189 single block inlinees; 37 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 208
       mov      qword ptr [rsp+0x568], r15
       mov      qword ptr [rsp+0x560], r14
       mov      qword ptr [rsp+0x558], r13
       mov      qword ptr [rsp+0x550], r12
       mov      qword ptr [rsp+0x548], rbx
       lea      rbp, [rsp+0xD0]
       xor      eax, eax
       mov      qword ptr [rbp-0x30], rax
       mov      qword ptr [rbp-0x38], rax
       mov      r15d, dword ptr [rbp+0x500]
       mov      ebx, dword ptr [rbp+0x508]
       mov      r13d, dword ptr [rbp+0x510]
       mov      r12d, dword ptr [rbp+0x518]
       mov      r14d, dword ptr [rbp+0x520]
       mov      r11d, dword ptr [rbp+0x42C]
       mov      r10d, dword ptr [rbp+0x428]
       mov      eax, dword ptr [rbp+0x424]
 
G_M000_IG02:                ;; offset=0x007C
       mov      r9, bword ptr [rbp+0x458]
       mov      bword ptr [rbp-0x60], r9
       mov      r8d, dword ptr [rbp+0x460]
       mov      dword ptr [rbp-0x3C], r8d
       mov      rcx, bword ptr [rbp+0x4D0]
       mov      bword ptr [rbp-0x68], rcx
       mov      edx, dword ptr [rbp+0x4D8]
       mov      dword ptr [rbp-0x40], edx
       mov      rsi, bword ptr [rbp+0x4E0]
       mov      bword ptr [rbp-0x70], rsi
       mov      edi, dword ptr [rbp+0x4E8]
       mov      dword ptr [rbp-0x44], edi
       mov      r9, bword ptr [rbp+0x4F0]
       mov      bword ptr [rbp-0x78], r9
       mov      r9d, dword ptr [rbp+0x4F8]
       mov      dword ptr [rbp-0x48], r9d
       mov      r8, bword ptr [rbp+0x468]
       mov      bword ptr [rbp-0x80], r8
       mov      r8d, dword ptr [rbp+0x470]
       mov      dword ptr [rbp-0x4C], r8d
       mov      r8, bword ptr [rbp+0x448]
       mov      bword ptr [rbp-0x88], r8
       mov      r8d, dword ptr [rbp+0x450]
       mov      dword ptr [rbp-0x50], r8d
       mov      r8, bword ptr [rbp+0x4B0]
       mov      bword ptr [rbp-0x90], r8
       mov      r8d, dword ptr [rbp+0x4B8]
       mov      dword ptr [rbp-0x54], r8d
       mov      r8, bword ptr [rbp+0x4C0]
       mov      bword ptr [rbp-0x98], r8
       mov      r8d, dword ptr [rbp+0x4C8]
       mov      dword ptr [rbp-0x58], r8d
       cmp      eax, r10d
       jl       G_M000_IG33
 
G_M000_IG03:                ;; offset=0x013A
       mov      rdi, bword ptr [rbp-0x78]
       mov      esi, dword ptr [rbp-0x48]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG09
 
G_M000_IG04:                ;; offset=0x014F
       mov      rdi, bword ptr [rbp-0x78]
       mov      esi, dword ptr [rbp-0x48]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG09
 
G_M000_IG05:                ;; offset=0x0164
       mov      rdi, bword ptr [rbp-0x88]
       mov      esi, dword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG09
 
G_M000_IG06:                ;; offset=0x017C
       mov      rdi, bword ptr [rbp-0x90]
       mov      esi, dword ptr [rbp-0x54]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG09
 
G_M000_IG07:                ;; offset=0x0190
       lea      rdi, [rsp]
       lea      rsi, [rbp+0x4B0]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       mov      dword ptr [rsp+0x10], ebx
       imul     r13d, r12d
       mov      dword ptr [rsp+0x18], r13d
       mov      dword ptr [rsp+0x20], r14d
       movzx    r8, byte  ptr [rbp+0x528]
       mov      dword ptr [rsp+0x28], r8d
       mov      r8, bword ptr [rbp-0x88]
       mov      r9d, dword ptr [rbp-0x50]
       mov      rdx, bword ptr [rbp-0x98]
       mov      ecx, dword ptr [rbp-0x58]
       mov      rdi, bword ptr [rbp-0x78]
       mov      esi, dword ptr [rbp-0x48]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG08:                ;; offset=0x01F2
       vzeroupper 
       add      rsp, 0x548
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG09:                ;; offset=0x0207
       xor      eax, eax
 
G_M000_IG10:                ;; offset=0x0209
       vzeroupper 
       add      rsp, 0x548
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG11:                ;; offset=0x021E
       jmp      G_M000_IG35
       align    [0 bytes for IG16]
 
G_M000_IG12:                ;; offset=0x0223
       inc      r10d
       cmp      r10d, 16
       mov      r12d, dword ptr [rbp+0x518]
       mov      r13d, dword ptr [rbp+0x510]
       jge      G_M000_IG19
 
G_M000_IG13:                ;; offset=0x023E
       xor      r9d, r9d
       cmp      r9d, ebx
       mov      dword ptr [rbp+0x510], r13d
       mov      dword ptr [rbp+0x518], r12d
       jge      SHORT G_M000_IG12
 
G_M000_IG14:                ;; offset=0x0254
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r11d, r10d
       imul     r11d, r15d
       mov      r12d, r11d
       imul     r12d, ebx
       movsxd   r12, r12d
       shl      r12, 2
       add      r12, rsi
       movsxd   r13, r9d
       lea      r13, [r12+4*r13]
       shl      r11d, 3
       movsxd   r11, r11d
       lea      r11, [rdi+4*r11]
       test     r15d, r15d
       jle      SHORT G_M000_IG17
 
G_M000_IG15:                ;; offset=0x02A3
       mov      r12d, r15d
 
G_M000_IG16:                ;; offset=0x02A6
       vmovups  ymm8, ymmword ptr [r13]
       vbroadcastss ymm9, dword ptr [r11]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       add      r13, rdx
       add      r11, 32
       dec      r12d
       jne      SHORT G_M000_IG16
 
G_M000_IG17:                ;; offset=0x030F
       mov      r11d, r10d
       imul     r11d, ebx
       add      r11d, r9d
       shl      r11d, 3
       movsxd   r11, r11d
       lea      r11, [r8+4*r11]
       vmovups  ymmword ptr [r11], ymm0
       vmovups  ymmword ptr [r11+0x20], ymm1
       vmovups  ymmword ptr [r11+0x40], ymm2
       vmovups  ymmword ptr [r11+0x60], ymm3
       vmovups  ymmword ptr [r11+0x80], ymm4
       vmovups  ymmword ptr [r11+0xA0], ymm5
       vmovups  ymmword ptr [r11+0xC0], ymm6
       vmovups  ymmword ptr [r11+0xE0], ymm7
       add      r9d, 8
       cmp      r9d, ebx
       jl       G_M000_IG14
       jmp      G_M000_IG12
 
G_M000_IG18:                ;; offset=0x0371
       mov      rdx, r8
       mov      ecx, r15d
       mov      r8d, ebx
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int)]
       mov      r8, qword ptr [rbp+0x408]
 
G_M000_IG19:                ;; offset=0x0387
       mov      ecx, dword ptr [rbp-0x44]
       mov      edi, ecx
       xor      rsi, rsi
       mov      bword ptr [rbp-0x38], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG20:                ;; offset=0x039B
       test     edi, edi
       je       SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x039F
       mov      r10, bword ptr [rbp-0x70]
       mov      rsi, r10
 
G_M000_IG22:                ;; offset=0x03A6
       mov      bword ptr [rbp-0x38], rsi
       xor      edi, edi
       cmp      ecx, 8
       jl       SHORT G_M000_IG24
       align    [0 bytes for IG23]
 
G_M000_IG23:                ;; offset=0x03B1
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
       jne      G_M000_IG56
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, ecx
       jle      SHORT G_M000_IG23
       align    [0 bytes for IG24]
 
G_M000_IG24:                ;; offset=0x03E3
       cmp      edi, ecx
       jl       G_M000_IG57
       xor      edi, edi
       mov      bword ptr [rbp-0x38], rdi
       mov      edi, 1
 
G_M000_IG25:                ;; offset=0x03F6
       xor      rsi, rsi
       mov      bword ptr [rbp-0x38], rsi
       test     edi, edi
       je       G_M000_IG09
 
G_M000_IG26:                ;; offset=0x0404
       cmp      r14d, 16
       je       G_M000_IG52
 
G_M000_IG27:                ;; offset=0x040E
       mov      r11d, dword ptr [rbp+0x424]
       mov      dword ptr [rsp], r11d
       mov      r9d, dword ptr [rbp+0x420]
       mov      dword ptr [rsp+0x08], r9d
       mov      rdi, r8
       mov      rsi, qword ptr [rbp+0x400]
       mov      edx, ebx
       mov      ecx, r13d
       mov      r8d, r12d
       mov      r9d, dword ptr [rbp+0x42C]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
 
G_M000_IG28:                ;; offset=0x0444
       xor      edx, edx
       mov      bword ptr [rbp+0x3F8], rdx
 
G_M000_IG29:                ;; offset=0x044D
       mov      bword ptr [rbp+0x3F0], rdx
 
G_M000_IG30:                ;; offset=0x0454
       mov      bword ptr [rbp+0x3E8], rdx
 
G_M000_IG31:                ;; offset=0x045B
       mov      bword ptr [rbp+0x3E0], rdx
       mov      r11d, dword ptr [rbp+0x424]
       add      r11d, 8
       mov      r8d, dword ptr [rbp+0x428]
       cmp      r11d, r8d
       mov      eax, r11d
       mov      r10d, r8d
       mov      rcx, bword ptr [rbp-0x68]
       jge      G_M000_IG03
 
G_M000_IG32:                ;; offset=0x0487
       mov      r11d, dword ptr [rbp+0x42C]
 
G_M000_IG33:                ;; offset=0x048E
       mov      dword ptr [rbp+0x428], r10d
       mov      r8d, r10d
       sub      r8d, eax
       cmp      r8d, 8
       jl       G_M000_IG11
 
G_M000_IG34:                ;; offset=0x04A5
       mov      r8d, 8
 
G_M000_IG35:                ;; offset=0x04AB
       mov      dword ptr [rbp+0x420], r8d
       mov      dword ptr [rbp+0x518], r12d
       mov      dword ptr [rsp], r12d
       mov      dword ptr [rbp+0x42C], r11d
       mov      dword ptr [rsp+0x08], r11d
       mov      dword ptr [rbp+0x424], eax
       mov      dword ptr [rsp+0x10], eax
       mov      dword ptr [rsp+0x18], r8d
       mov      rdi, bword ptr [rbp-0x80]
       mov      esi, dword ptr [rbp-0x4C]
       mov      rdx, rcx
       mov      ecx, dword ptr [rbp-0x40]
       mov      r8d, r15d
       mov      dword ptr [rbp+0x510], r13d
       mov      r9d, r13d
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      eax, dword ptr [rbp-0x40]
       mov      edi, eax
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG36:                ;; offset=0x050C
       test     edi, edi
       mov      r12d, dword ptr [rbp+0x518]
       mov      r13d, dword ptr [rbp+0x510]
       je       SHORT G_M000_IG38
 
G_M000_IG37:                ;; offset=0x051E
       mov      r9, bword ptr [rbp-0x68]
       mov      rsi, r9
 
G_M000_IG38:                ;; offset=0x0525
       mov      bword ptr [rbp-0x30], rsi
       xor      edi, edi
       cmp      eax, 8
       jl       SHORT G_M000_IG40
       align    [0 bytes for IG39]
 
G_M000_IG39:                ;; offset=0x0530
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
       jne      G_M000_IG54
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, eax
       jle      SHORT G_M000_IG39
       align    [0 bytes for IG40]
 
G_M000_IG40:                ;; offset=0x0562
       cmp      edi, eax
       jl       G_M000_IG55
       xor      edi, edi
       mov      bword ptr [rbp-0x30], rdi
       mov      edi, 1
 
G_M000_IG41:                ;; offset=0x0575
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       test     edi, edi
       je       G_M000_IG09
 
G_M000_IG42:                ;; offset=0x0583
       xor      rdi, rdi
       test     eax, eax
       je       SHORT G_M000_IG44
 
G_M000_IG43:                ;; offset=0x0589
       mov      r9, bword ptr [rbp-0x68]
       mov      rdi, r9
 
G_M000_IG44:                ;; offset=0x0590
       mov      bword ptr [rbp+0x3F8], rdi
       xor      rsi, rsi
       cmp      dword ptr [rbp-0x3C], 0
       je       SHORT G_M000_IG46
 
G_M000_IG45:                ;; offset=0x059F
       mov      r11, bword ptr [rbp-0x60]
       mov      rsi, r11
 
G_M000_IG46:                ;; offset=0x05A6
       mov      bword ptr [rbp+0x3F0], rsi
       xor      r8, r8
       cmp      dword ptr [rbp-0x44], 0
       je       SHORT G_M000_IG48
 
G_M000_IG47:                ;; offset=0x05B6
       mov      rdx, bword ptr [rbp-0x70]
       mov      r8, rdx
 
G_M000_IG48:                ;; offset=0x05BD
       mov      bword ptr [rbp+0x3E8], r8
       mov      qword ptr [rbp+0x408], r8
       xor      r11, r11
       cmp      dword ptr [rbp-0x48], 0
       je       SHORT G_M000_IG50
 
G_M000_IG49:                ;; offset=0x05D4
       mov      r11, bword ptr [rbp-0x78]
 
G_M000_IG50:                ;; offset=0x05D8
       mov      bword ptr [rbp+0x3E0], r11
       mov      qword ptr [rbp+0x400], r11
       cmp      r14d, 16
       je       G_M000_IG18
 
G_M000_IG51:                ;; offset=0x05F0
       xor      r10d, r10d
       movsxd   rdx, ebx
       shl      rdx, 2
       jmp      G_M000_IG13
       align    [7 bytes for IG53]
 
G_M000_IG52:                ;; offset=0x0606
       xor      edx, edx
       test     ebx, ebx
       jle      G_M000_IG28
 
G_M000_IG53:                ;; offset=0x0610
       mov      r9d, dword ptr [rbp+0x420]
       test     r9d, r9d
       jg       SHORT G_M000_IG59
       add      edx, 16
       cmp      edx, ebx
       jl       SHORT G_M000_IG53
       jmp      G_M000_IG28
 
G_M000_IG54:                ;; offset=0x0628
       xor      edi, edi
       jmp      G_M000_IG41
 
G_M000_IG55:                ;; offset=0x062F
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      ecx, 0x7F800000
       andn     edx, edx, ecx
       je       SHORT G_M000_IG54
       inc      edi
       jmp      G_M000_IG40
 
G_M000_IG56:                ;; offset=0x0648
       xor      edi, edi
       jmp      G_M000_IG25
 
G_M000_IG57:                ;; offset=0x064F
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      r9d, 0x7F800000
       andn     edx, edx, r9d
       je       SHORT G_M000_IG58
       inc      edi
       mov      ecx, dword ptr [rbp-0x44]
       jmp      G_M000_IG24
 
G_M000_IG58:                ;; offset=0x066C
       mov      ecx, dword ptr [rbp-0x44]
       jmp      SHORT G_M000_IG56
 
G_M000_IG59:                ;; offset=0x0671
       mov      r11d, dword ptr [rbp+0x424]
       lea      eax, [8*rdx]
       cdqe     
       lea      rcx, [r8+4*rax]
       shl      ebx, 3
       mov      r13d, dword ptr [rbp+0x42C]
       mov      eax, r11d
       cdq      
       idiv     edx:eax, r13d
       mov      eax, r11d
       cdq      
       idiv     edx:eax, r13d
       cmp      byte  ptr [rcx], cl
       shl      ebx, 2
       movsxd   rax, ebx
       movsx    rax, byte  ptr [rcx+4*rax]
       call     CORINFO_HELP_THROW_PLATFORM_NOT_SUPPORTED
       int3     
 
RWD00  	dd	7F800000h		;       inf

; Total bytes of code 1712

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 416
       lea      rbp, [rsp+0x1A0]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x190], xmm8
       vmovdqa  xmmword ptr [rbp-0x180], xmm8
       mov      rax, -288
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
 
G_M000_IG02:                ;; offset=0x005E
       mov      dword ptr [rbp-0x1A0], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x4C], eax
       jmp      G_M000_IG12
 
G_M000_IG03:                ;; offset=0x0072
       xor      eax, eax
       mov      dword ptr [rbp-0x50], eax
       jmp      G_M000_IG09
 
G_M000_IG04:                ;; offset=0x007C
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [rbp-0x70], ymm0
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
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       imul     eax, dword ptr [rbp-0x48]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0x38]
       movsxd   rcx, dword ptr [rbp-0x50]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x158], rax
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x44]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x30]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x160], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x164], eax
       jmp      G_M000_IG06
 
G_M000_IG05:                ;; offset=0x0125
       mov      rdi, 0x76DAAF74AFC8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rax, qword ptr [rbp-0x158]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x190], ymm0
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax]
       vmovups  ymm1, ymmword ptr [rbp-0x70]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x70], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x04]
       vmovups  ymm1, ymmword ptr [rbp-0x90]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x90], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x08]
       vmovups  ymm1, ymmword ptr [rbp-0xB0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xB0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x0C]
       vmovups  ymm1, ymmword ptr [rbp-0xD0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xD0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x10]
       vmovups  ymm1, ymmword ptr [rbp-0xF0]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0xF0], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x14]
       vmovups  ymm1, ymmword ptr [rbp-0x110]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x110], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x18]
       vmovups  ymm1, ymmword ptr [rbp-0x130]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x130], ymm1
       mov      rax, qword ptr [rbp-0x160]
       vbroadcastss ymm0, dword ptr [rax+0x1C]
       vmovups  ymm1, ymmword ptr [rbp-0x150]
       vfmadd231ps ymm1, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x150], ymm1
       movsxd   rax, dword ptr [rbp-0x48]
       mov      rcx, qword ptr [rbp-0x158]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x158], rax
       mov      rax, qword ptr [rbp-0x160]
       add      rax, 32
       mov      qword ptr [rbp-0x160], rax
       mov      eax, dword ptr [rbp-0x164]
       inc      eax
       mov      dword ptr [rbp-0x164], eax
 
G_M000_IG06:                ;; offset=0x02A6
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x02BD
       lea      rdi, [rbp-0x1A0]
       mov      esi, 320
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x02CE
       mov      eax, dword ptr [rbp-0x164]
       cmp      eax, dword ptr [rbp-0x44]
       jl       G_M000_IG05
       mov      rdi, 0x76DAAF74AFCC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       imul     eax, dword ptr [rbp-0x48]
       add      eax, dword ptr [rbp-0x50]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x40]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0x198], rax
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vmovups  ymmword ptr [rax], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x90]
       vmovups  ymmword ptr [rax+0x20], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xB0]
       vmovups  ymmword ptr [rax+0x40], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xD0]
       vmovups  ymmword ptr [rax+0x60], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0xF0]
       vmovups  ymmword ptr [rax+0x80], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x110]
       vmovups  ymmword ptr [rax+0xA0], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x130]
       vmovups  ymmword ptr [rax+0xC0], ymm0
       mov      rax, qword ptr [rbp-0x198]
       vmovups  ymm0, ymmword ptr [rbp-0x150]
       vmovups  ymmword ptr [rax+0xE0], ymm0
       mov      eax, dword ptr [rbp-0x50]
       add      eax, 8
       mov      dword ptr [rbp-0x50], eax
 
G_M000_IG09:                ;; offset=0x03BB
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x03D2
       lea      rdi, [rbp-0x1A0]
       mov      esi, 447
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG11:                ;; offset=0x03E3
       mov      eax, dword ptr [rbp-0x50]
       cmp      eax, dword ptr [rbp-0x48]
       jl       G_M000_IG04
       mov      rdi, 0x76DAAF74AFD0
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x4C]
       inc      eax
       mov      dword ptr [rbp-0x4C], eax
 
G_M000_IG12:                ;; offset=0x0406
       mov      eax, dword ptr [rbp-0x1A0]
       dec      eax
       mov      dword ptr [rbp-0x1A0], eax
       cmp      dword ptr [rbp-0x1A0], 0
       jg       SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x041D
       lea      rdi, [rbp-0x1A0]
       mov      esi, 459
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG14:                ;; offset=0x042E
       cmp      dword ptr [rbp-0x4C], 16
       jl       G_M000_IG03
       mov      rdi, 0x76DAAF74AFD4
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG15:                ;; offset=0x0448
       vzeroupper 
       add      rsp, 416
       pop      rbp
       ret      
 
; Total bytes of code 1108

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x140
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 6

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 16
       mov      qword ptr [rsp+0x1B8], r15
       mov      qword ptr [rsp+0x1B0], rbx
       lea      rbp, [rsp+0x10]
       mov      rdi, qword ptr [rbp+0x180]
       mov      rsi, qword ptr [rbp+0x178]
       mov      rdx, qword ptr [rbp+0x170]
       mov      ecx, dword ptr [rbp+0x16C]
       mov      eax, dword ptr [rbp+0x168]
       mov      ebx, dword ptr [rbp+0x164]
       mov      r11d, dword ptr [rbp+0x160]
       vmovups  ymm0, ymmword ptr [rbp+0x140]
       vmovups  ymm1, ymmword ptr [rbp+0x120]
       vmovups  ymm2, ymmword ptr [rbp+0x100]
       vmovups  ymm3, ymmword ptr [rbp+0xE0]
       vmovups  ymm4, ymmword ptr [rbp+0xC0]
       vmovups  ymm5, ymmword ptr [rbp+0xA0]
       vmovups  ymm6, ymmword ptr [rbp+0x80]
       vmovups  ymm7, ymmword ptr [rbp+0x60]
       mov      r10, qword ptr [rbp+0x58]
       mov      r8, qword ptr [rbp+0x50]
       mov      r9d, dword ptr [rbp+0x4C]
 
G_M000_IG02:                ;; offset=0x0095
       jmp      G_M000_IG05
       align    [8 bytes for IG07]
 
G_M000_IG03:                ;; offset=0x00A2
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r8d, ebx
       imul     r8d, ecx
       mov      r9d, r8d
       imul     r9d, eax
       movsxd   r9, r9d
       shl      r9, 2
       add      r9, rsi
       movsxd   r10, r11d
       lea      r10, [r9+4*r10]
       shl      r8d, 3
       movsxd   r8, r8d
       lea      r8, [rdi+4*r8]
       xor      r9d, r9d
       cmp      r9d, ecx
       jge      SHORT G_M000_IG06
 
G_M000_IG04:                ;; offset=0x00F4
       vmovups  ymm8, ymmword ptr [r10]
       vbroadcastss ymm9, dword ptr [r8]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r8+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       movsxd   r15, eax
       lea      r10, [r10+4*r15]
       add      r8, 32
       inc      r9d
 
G_M000_IG05:                ;; offset=0x015E
       cmp      r9d, ecx
       jl       SHORT G_M000_IG04
 
G_M000_IG06:                ;; offset=0x0163
       mov      r10d, ebx
       imul     r10d, eax
       add      r10d, r11d
       shl      r10d, 3
       movsxd   r8, r10d
       lea      r9, [rdx+4*r8]
       vmovups  ymmword ptr [r9], ymm0
       vmovups  ymmword ptr [r9+0x20], ymm1
       vmovups  ymmword ptr [r9+0x40], ymm2
       vmovups  ymmword ptr [r9+0x60], ymm3
       vmovups  ymmword ptr [r9+0x80], ymm4
       vmovups  ymmword ptr [r9+0xA0], ymm5
       vmovups  ymmword ptr [r9+0xC0], ymm6
       vmovups  ymmword ptr [r9+0xE0], ymm7
       add      r11d, 8
       cmp      r11d, eax
       jl       G_M000_IG03
 
G_M000_IG07:                ;; offset=0x01C0
       inc      ebx
       cmp      ebx, 16
       jge      SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x01C7
       xor      r11d, r11d
       test     eax, eax
       jg       G_M000_IG03
       jmp      SHORT G_M000_IG07
 
G_M000_IG09:                ;; offset=0x01D4
       vzeroupper 
       add      rsp, 432
       pop      rbx
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 483

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
       mov      rdi, 0x76DAAF74B018
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
       mov      rdi, 0x76DAAF74B01C
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x44]
       lea      eax, [8*rax+0x08]
       cdqe     
       mov      rcx, qword ptr [rbp-0x1D8]
       vmovups  ymm0, ymmword ptr [rbp-0x1D0]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
 
G_M000_IG09:                ;; offset=0x03FC
       mov      rdi, 0x76DAAF74B020
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
       mov      rdi, 0x76DAAF74B024
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
       mov      rdi, 0x76DAAF74B028
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG16:                ;; offset=0x04A3
       vzeroupper 
       add      rsp, 576
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x04AF
       mov      rdi, 0x76DAAF74B02C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG09
 
; Total bytes of code 1219

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 5

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     rbx
       lea      rbp, [rsp+0x10]
 
G_M000_IG02:                ;; offset=0x0009
       xor      eax, eax
       movsxd   r9, r8d
       shl      r9, 2
       jmp      SHORT G_M000_IG04
       align    [0 bytes for IG07]
 
G_M000_IG03:                ;; offset=0x0014
       inc      eax
       cmp      eax, 16
       jge      G_M000_IG09
 
G_M000_IG04:                ;; offset=0x001F
       xor      r10d, r10d
       cmp      r10d, r8d
       jge      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x0027
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r11d, eax
       imul     r11d, ecx
       mov      ebx, r11d
       imul     ebx, r8d
       movsxd   rbx, ebx
       shl      rbx, 2
       add      rbx, rsi
       movsxd   r15, r10d
       lea      rbx, [rbx+4*r15]
       shl      r11d, 3
       movsxd   r11, r11d
       lea      r11, [rdi+4*r11]
       test     ecx, ecx
       jle      SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0075
       mov      r15d, ecx
 
G_M000_IG07:                ;; offset=0x0078
       vmovups  ymm8, ymmword ptr [rbx]
       vbroadcastss ymm9, dword ptr [r11]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r11+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       add      rbx, r9
       add      r11, 32
       dec      r15d
       jne      SHORT G_M000_IG07
 
G_M000_IG08:                ;; offset=0x00DF
       mov      r11d, eax
       imul     r11d, r8d
       add      r11d, r10d
       shl      r11d, 3
       movsxd   r11, r11d
       lea      r11, [rdx+4*r11]
       vmovups  ymmword ptr [r11], ymm0
       vmovups  ymmword ptr [r11+0x20], ymm1
       vmovups  ymmword ptr [r11+0x40], ymm2
       vmovups  ymmword ptr [r11+0x60], ymm3
       vmovups  ymmword ptr [r11+0x80], ymm4
       vmovups  ymmword ptr [r11+0xA0], ymm5
       vmovups  ymmword ptr [r11+0xC0], ymm6
       vmovups  ymmword ptr [r11+0xE0], ymm7
       add      r10d, 8
       cmp      r10d, r8d
       jl       G_M000_IG05
       jmp      G_M000_IG03
 
G_M000_IG09:                ;; offset=0x0141
       vzeroupper 
       pop      rbx
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 329

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 32704

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

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 288
       lea      rbp, [rsp+0x120]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x110], xmm8
       mov      rax, -192
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x20], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x40], rax
       mov      bword ptr [rbp-0x38], rdi
       mov      qword ptr [rbp-0x30], rsi
 
G_M000_IG02:                ;; offset=0x004B
       mov      dword ptr [rbp-0x118], 0x3E8
       xor      eax, eax
       mov      dword ptr [rbp-0x3C], eax
       lea      rdi, [rbp-0x38]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0x50], rax
       mov      rax, bword ptr [rbp-0x50]
       mov      qword ptr [rbp-0x120], rax
       mov      rax, qword ptr [rbp-0x120]
       mov      qword ptr [rbp-0x48], rax
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       vmovups  ymmword ptr [rbp-0xF0], ymm0
       vbroadcastss ymm0, dword ptr [reloc @RWD04]
       vmovups  ymmword ptr [rbp-0x110], ymm0
       jmp      SHORT G_M000_IG06
 
G_M000_IG03:                ;; offset=0x00A2
       movsxd   rax, dword ptr [rbp-0x3C]
       mov      rcx, qword ptr [rbp-0x48]
       vmovups  ymm0, ymmword ptr [rcx+4*rax]
       vpand    ymm0, ymm0, ymmword ptr [rbp-0xF0]
       vcmpgtps ymm0, ymm0, ymmword ptr [rbp-0x110]
       vmovmskps rax, ymm0
       test     eax, eax
       je       SHORT G_M000_IG05
       mov      rdi, 0x76DAAF750FC8
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG04:                ;; offset=0x00D9
       vzeroupper 
       add      rsp, 288
       pop      rbp
       ret      
 
G_M000_IG05:                ;; offset=0x00E5
       mov      rdi, 0x76DAAF750FCC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       add      eax, 8
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG06:                ;; offset=0x00FD
       mov      eax, dword ptr [rbp-0x118]
       dec      eax
       mov      dword ptr [rbp-0x118], eax
       cmp      dword ptr [rbp-0x118], 0
       jg       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x0114
       lea      rdi, [rbp-0x118]
       mov      esi, 191
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG08:                ;; offset=0x0125
       mov      eax, dword ptr [rbp-0x30]
       add      eax, -8
       cmp      dword ptr [rbp-0x3C], eax
       jle      G_M000_IG03
 
G_M000_IG09:                ;; offset=0x0134
       mov      rdi, 0x76DAAF750FD0
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
       mov      bword ptr [rbp-0x50], rax
       jmp      SHORT G_M000_IG13
 
G_M000_IG10:                ;; offset=0x014B
       mov      eax, dword ptr [rbp-0x30]
       cmp      dword ptr [rbp-0x3C], eax
       jae      G_M000_IG17
       mov      eax, dword ptr [rbp-0x3C]
       mov      rcx, bword ptr [rbp-0x38]
       vmovss   xmm0, dword ptr [rcx+4*rax]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       jbe      SHORT G_M000_IG12
       mov      rdi, 0x76DAAF750FD4
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG11:                ;; offset=0x0186
       vzeroupper 
       add      rsp, 288
       pop      rbp
       ret      
 
G_M000_IG12:                ;; offset=0x0192
       mov      rdi, 0x76DAAF750FD8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x3C]
       inc      eax
       mov      dword ptr [rbp-0x3C], eax
 
G_M000_IG13:                ;; offset=0x01A9
       mov      eax, dword ptr [rbp-0x118]
       dec      eax
       mov      dword ptr [rbp-0x118], eax
       cmp      dword ptr [rbp-0x118], 0
       jg       SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x01C0
       lea      rdi, [rbp-0x118]
       mov      esi, 235
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG15:                ;; offset=0x01D1
       mov      eax, dword ptr [rbp-0x3C]
       cmp      eax, dword ptr [rbp-0x30]
       jl       G_M000_IG10
       mov      rdi, 0x76DAAF750FDC
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, 1
 
G_M000_IG16:                ;; offset=0x01F1
       vzeroupper 
       add      rsp, 288
       pop      rbp
       ret      
 
G_M000_IG17:                ;; offset=0x01FD
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	7E7FFFFFh		; 8.50706e+37
RWD08  	dd	00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 515

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0xbf
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 59
; 1 inlinees with PGO data; 0 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       mov      rbp, rsp
       mov      eax, dword ptr [rbp+0xF4]
       mov      rcx, qword ptr [rbp+0xE8]
       vmovups  ymm0, ymmword ptr [rbp+0x40]
       vmovups  ymm1, ymmword ptr [rbp+0x20]
 
G_M000_IG02:                ;; offset=0x001F
       mov      edx, dword ptr [rbp+0x100]
       lea      edi, [rdx-0x08]
       cmp      eax, edi
       jg       SHORT G_M000_IG05
       align    [4 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0030
       movsxd   rsi, eax
       vpand    ymm2, ymm0, ymmword ptr [rcx+4*rsi]
       vcmpgtps ymm2, ymm2, ymm1
       vmovmskps rsi, ymm2
       test     esi, esi
       jne      SHORT G_M000_IG11
 
G_M000_IG04:                ;; offset=0x0045
       add      eax, 8
       cmp      eax, edi
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x004C
       xor      ecx, ecx
       mov      bword ptr [rbp+0xE0], rcx
       cmp      eax, edx
       jl       SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0059
       mov      eax, 1
 
G_M000_IG07:                ;; offset=0x005E
       vzeroupper 
       add      rsp, 304
       pop      rbp
       ret      
 
G_M000_IG08:                ;; offset=0x006A
       test     eax, eax
       jl       SHORT G_M000_IG13
 
G_M000_IG09:                ;; offset=0x006E
       mov      rcx, bword ptr [rbp+0xF8]
       mov      edi, eax
       vmovss   xmm0, dword ptr [rcx+4*rdi]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD00]
       vucomiss xmm0, dword ptr [reloc @RWD16]
       ja       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x008E
       inc      eax
       cmp      eax, edx
       jl       SHORT G_M000_IG09
       jmp      SHORT G_M000_IG06
 
G_M000_IG11:                ;; offset=0x0096
       xor      eax, eax
 
G_M000_IG12:                ;; offset=0x0098
       vzeroupper 
       add      rsp, 304
       pop      rbp
       ret      
 
G_M000_IG13:                ;; offset=0x00A4
       cmp      eax, edx
       jae      SHORT G_M000_IG15
       mov      rcx, bword ptr [rbp+0xF8]
       mov      edi, eax
       vmovss   xmm0, dword ptr [rcx+4*rdi]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD00]
       vucomiss xmm0, dword ptr [reloc @RWD16]
       ja       SHORT G_M000_IG11
 
G_M000_IG14:                ;; offset=0x00C8
       inc      eax
       cmp      eax, edx
       jl       SHORT G_M000_IG13
       jmp      SHORT G_M000_IG06
 
G_M000_IG15:                ;; offset=0x00D0
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh
RWD16  	dd	7E7FFFFFh		; 8.50706e+37

; Total bytes of code 214

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       mov      rdi, 0x76DAAF8C6690
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG04:                ;; offset=0x0073
       mov      rdi, 0x76DAAF8C6694
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
 
G_M000_IG05:                ;; offset=0x0084
       add      rsp, 80
       pop      rbp
       ret      
 
G_M000_IG06:                ;; offset=0x008A
       mov      rdi, 0x76DAAF8C6698
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG04
 
G_M000_IG07:                ;; offset=0x009B
       mov      rdi, 0x76DAAF8C669C
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
       mov      rdi, 0x76DAAF8C66A0
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
       mov      rdi, 0x76DAAF8C66A8
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
       mov      rdi, 0x76DAAF8C66A4
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
       mov      dword ptr [rbp-0x44], eax
       lea      rax, G_M000_IG11
 
G_M000_IG15:                ;; offset=0x017E
       add      rsp, 8
       ret      
 
; Total bytes of code 387

; Assembly listing for method KernelAccess:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       mov      rax, 0x76D2C9800198
       mov      rax, gword ptr [rax]
       mov      gword ptr [rbp-0x28], rax
       mov      rdi, gword ptr [rbp-0x28]
       mov      rsi, 0x76DAAF8CAC48
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
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       mov      rax, 0x76D2C98001A8
       mov      rax, gword ptr [rax]
       mov      gword ptr [rbp-0x38], rax
       mov      rdi, gword ptr [rbp-0x38]
       mov      rsi, 0x76DAAF8CAD80
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
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Instrumented Tier0 code
; rbp based frame
; partially interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x490
       lea      rbp, [rsp+0x490]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x430], xmm8
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
 
G_M000_IG02:                ;; offset=0x005B
       mov      dword ptr [rbp-0x420], 0x3E8
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
       mov      rdi, 0x76DAAF74AE60
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG97
 
G_M000_IG03:                ;; offset=0x00BE
       mov      eax, dword ptr [rbp+0x60]
       imul     eax, dword ptr [rbp+0x70]
       jo       G_M000_IG103
       imul     eax, dword ptr [rbp+0x78]
       jo       G_M000_IG103
       cmp      dword ptr [rbp-0x30], eax
       jne      G_M000_IG07
       imul     eax, dword ptr [rbp+0x60], 16
       jo       G_M000_IG103
       imul     eax, dword ptr [rbp+0x68]
       jo       G_M000_IG103
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
       mov      rdi, 0x76DAAF74AE64
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG04:                ;; offset=0x0128
       cmp      dword ptr [rbp+0x18], 0
       je       SHORT G_M000_IG05
       mov      eax, dword ptr [rbp+0x18]
       cmp      eax, dword ptr [rbp-0x70]
       jne      G_M000_IG10
       mov      rdi, 0x76DAAF74AE68
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG05:                ;; offset=0x0149
       mov      eax, dword ptr [rbp+0x38]
       cmp      eax, dword ptr [rbp-0x60]
       jl       G_M000_IG09
       mov      eax, dword ptr [rbp+0x48]
       cmp      eax, dword ptr [rbp-0x68]
       jl       SHORT G_M000_IG08
       mov      eax, dword ptr [rbp+0x58]
       cmp      eax, dword ptr [rbp-0x70]
       jge      G_M000_IG14
 
G_M000_IG06:                ;; offset=0x0169
       mov      rdi, 0x76DAAF74AE6C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG07:                ;; offset=0x0178
       mov      rdi, 0x76DAAEC7CD28
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0xC8], rax
       mov      edi, 0x102B4
       mov      rsi, 0x76DAAF4A9F30
       call     [CORINFO_HELP_STRCNS]
       mov      gword ptr [rbp-0x428], rax
       mov      rsi, gword ptr [rbp-0x428]
       mov      rdi, gword ptr [rbp-0xC8]
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, gword ptr [rbp-0xC8]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG08:                ;; offset=0x01CB
       mov      rdi, 0x76DAAF74AE70
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG07
 
G_M000_IG09:                ;; offset=0x01DC
       mov      rdi, 0x76DAAF74AE74
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG07
 
G_M000_IG10:                ;; offset=0x01ED
       mov      rdi, 0x76DAAF74AE78
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG11:                ;; offset=0x0201
       mov      rdi, 0x76DAAF74AE7C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG12:                ;; offset=0x0215
       mov      rdi, 0x76DAAF74AE80
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG13:                ;; offset=0x0229
       mov      rdi, 0x76DAAF74AE84
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG14:                ;; offset=0x023D
       lea      rdi, [rbp+0x30]
       mov      edx, dword ptr [rbp-0x60]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xD8], rax
       mov      qword ptr [rbp-0xD0], rdx
 
G_M000_IG15:                ;; offset=0x025A
       vmovdqu  xmm0, xmmword ptr [rbp-0xD8]
       vmovdqu  xmmword ptr [rbp+0x30], xmm0
 
G_M000_IG16:                ;; offset=0x0267
       lea      rdi, [rbp+0x40]
       mov      edx, dword ptr [rbp-0x68]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xE8], rax
       mov      qword ptr [rbp-0xE0], rdx
 
G_M000_IG17:                ;; offset=0x0284
       vmovdqu  xmm0, xmmword ptr [rbp-0xE8]
       vmovdqu  xmmword ptr [rbp+0x40], xmm0
 
G_M000_IG18:                ;; offset=0x0291
       lea      rdi, [rbp+0x50]
       mov      edx, dword ptr [rbp-0x70]
       xor      esi, esi
       call     [System.Span`1[float]:Slice(int,int):System.Span`1[float]:this]
       mov      bword ptr [rbp-0xF8], rax
       mov      qword ptr [rbp-0xF0], rdx
 
G_M000_IG19:                ;; offset=0x02AE
       vmovdqu  xmm0, xmmword ptr [rbp-0xF8]
       vmovdqu  xmmword ptr [rbp+0x50], xmm0
 
G_M000_IG20:                ;; offset=0x02BB
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu  xmmword ptr [rbp-0x118], xmm0
 
G_M000_IG21:                ;; offset=0x02C8
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
 
G_M000_IG22:                ;; offset=0x030E
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu  xmmword ptr [rbp-0x140], xmm0
 
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
       vmovdqu  xmmword ptr [rbp-0x160], xmm0
 
G_M000_IG25:                ;; offset=0x036E
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
 
G_M000_IG26:                ;; offset=0x03B4
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu  xmmword ptr [rbp-0x180], xmm0
 
G_M000_IG27:                ;; offset=0x03C1
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
 
G_M000_IG28:                ;; offset=0x0407
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu  xmmword ptr [rbp-0x1A0], xmm0
 
G_M000_IG29:                ;; offset=0x0414
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
 
G_M000_IG30:                ;; offset=0x045A
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu  xmmword ptr [rbp-0x1C0], xmm0
 
G_M000_IG31:                ;; offset=0x0467
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
 
G_M000_IG32:                ;; offset=0x04AD
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu  xmmword ptr [rbp-0x1E0], xmm0
 
G_M000_IG33:                ;; offset=0x04BA
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
 
G_M000_IG34:                ;; offset=0x0500
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu  xmmword ptr [rbp-0x200], xmm0
 
G_M000_IG35:                ;; offset=0x050D
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
 
G_M000_IG36:                ;; offset=0x0553
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu  xmmword ptr [rbp-0x220], xmm0
 
G_M000_IG37:                ;; offset=0x0560
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
 
G_M000_IG38:                ;; offset=0x05A6
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu  xmmword ptr [rbp-0x240], xmm0
 
G_M000_IG39:                ;; offset=0x05B3
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
 
G_M000_IG40:                ;; offset=0x05F9
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu  xmmword ptr [rbp-0x260], xmm0
 
G_M000_IG41:                ;; offset=0x0606
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
 
G_M000_IG42:                ;; offset=0x064C
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu  xmmword ptr [rbp-0x280], xmm0
 
G_M000_IG43:                ;; offset=0x0659
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
 
G_M000_IG44:                ;; offset=0x069F
       vmovdqu  xmm0, xmmword ptr [rbp-0x38]
       vmovdqu  xmmword ptr [rbp-0x2A0], xmm0
 
G_M000_IG45:                ;; offset=0x06AC
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
 
G_M000_IG46:                ;; offset=0x06F2
       vmovdqu  xmm0, xmmword ptr [rbp-0x48]
       vmovdqu  xmmword ptr [rbp-0x2C0], xmm0
 
G_M000_IG47:                ;; offset=0x06FF
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
 
G_M000_IG48:                ;; offset=0x0745
       vmovdqu  xmm0, xmmword ptr [rbp-0x58]
       vmovdqu  xmmword ptr [rbp-0x2E0], xmm0
 
G_M000_IG49:                ;; offset=0x0752
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
 
G_M000_IG50:                ;; offset=0x0798
       vmovdqu  xmm0, xmmword ptr [rbp+0x10]
       vmovdqu  xmmword ptr [rbp-0x300], xmm0
 
G_M000_IG51:                ;; offset=0x07A5
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
 
G_M000_IG52:                ;; offset=0x08E7
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
 
G_M000_IG53:                ;; offset=0x0A2F
       test     eax, eax
       je       G_M000_IG76
 
G_M000_IG54:                ;; offset=0x0A37
       mov      rdi, 0x76DAAF74AE88
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG55:                ;; offset=0x0A46
       mov      rdi, 0x76DAAEC7CD28
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0x120], rax
       mov      edi, 0x102E2
       mov      rsi, 0x76DAAF4A9F30
       call     [CORINFO_HELP_STRCNS]
       mov      gword ptr [rbp-0x430], rax
       mov      rsi, gword ptr [rbp-0x430]
       mov      rdi, gword ptr [rbp-0x120]
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, gword ptr [rbp-0x120]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG56:                ;; offset=0x0A99
       mov      rdi, 0x76DAAF74AE8C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG55
 
G_M000_IG57:                ;; offset=0x0AAA
       mov      rdi, 0x76DAAF74AE90
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG55
 
G_M000_IG58:                ;; offset=0x0ABB
       mov      rdi, 0x76DAAF74AE94
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG59:                ;; offset=0x0ACF
       mov      rdi, 0x76DAAF74AE98
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG60:                ;; offset=0x0AE3
       mov      rdi, 0x76DAAF74AE9C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG61:                ;; offset=0x0AF7
       mov      rdi, 0x76DAAF74AEA0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG62:                ;; offset=0x0B0B
       mov      rdi, 0x76DAAF74AEA4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG63:                ;; offset=0x0B1F
       mov      rdi, 0x76DAAF74AEA8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG64:                ;; offset=0x0B33
       mov      rdi, 0x76DAAF74AEAC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG65:                ;; offset=0x0B47
       mov      rdi, 0x76DAAF74AEB0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG66:                ;; offset=0x0B5B
       mov      rdi, 0x76DAAF74AEB4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG67:                ;; offset=0x0B6F
       mov      rdi, 0x76DAAF74AEB8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG68:                ;; offset=0x0B83
       mov      rdi, 0x76DAAF74AEBC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG69:                ;; offset=0x0B97
       mov      rdi, 0x76DAAF74AEC0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG70:                ;; offset=0x0BAB
       mov      rdi, 0x76DAAF74AEC4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG71:                ;; offset=0x0BBF
       mov      rdi, 0x76DAAF74AEC8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG72:                ;; offset=0x0BD3
       mov      rdi, 0x76DAAF74AECC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG73:                ;; offset=0x0BE7
       mov      rdi, 0x76DAAF74AED0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG74:                ;; offset=0x0BFB
       mov      rdi, 0x76DAAF74AED4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG75:                ;; offset=0x0C0F
       mov      rdi, 0x76DAAF74AED8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG76:                ;; offset=0x0C23
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG77
       mov      rdi, 0x76DAAF74AEDC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG78
 
G_M000_IG77:                ;; offset=0x0C3D
       cmp      dword ptr [rbp+0x80], 8
       jne      SHORT G_M000_IG79
       mov      rdi, 0x76DAAF74AEE0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG79
 
G_M000_IG78:                ;; offset=0x0C57
       mov      rdi, 0x76DAAF34A470
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0x418], rax
       mov      rdi, gword ptr [rbp-0x418]
       call     [System.PlatformNotSupportedException:.ctor():this]
       mov      rdi, gword ptr [rbp-0x418]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG79:                ;; offset=0x0C87
       mov      rdi, bword ptr [rbp-0x38]
       mov      rsi, qword ptr [rbp-0x30]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG81
       mov      rdi, bword ptr [rbp-0x48]
       mov      rsi, qword ptr [rbp-0x40]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG83
       mov      rdi, bword ptr [rbp-0x58]
       mov      rsi, qword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG82
       mov      rdi, bword ptr [rbp+0x10]
       mov      rsi, qword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG84
 
G_M000_IG80:                ;; offset=0x0CCF
       mov      rdi, 0x76DAAF74AEE4
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG81:                ;; offset=0x0CDE
       mov      rdi, 0x76DAAF74AEE8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG97
 
G_M000_IG82:                ;; offset=0x0CF2
       mov      rdi, 0x76DAAF74AEEC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG81
 
G_M000_IG83:                ;; offset=0x0D03
       mov      rdi, 0x76DAAF74AEF0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG81
 
G_M000_IG84:                ;; offset=0x0D14
       mov      eax, dword ptr [rbp+0x78]
       inc      eax
       mov      dword ptr [rbp-0x454], eax
       mov      eax, dword ptr [rbp-0x454]
       shr      eax, 31
       add      eax, dword ptr [rbp-0x454]
       sar      eax, 1
       mov      dword ptr [rbp-0x74], eax
       mov      eax, dword ptr [rbp+0x70]
       add      eax, 1
       jo       G_M000_IG103
       mov      dword ptr [rbp-0x458], eax
       mov      eax, dword ptr [rbp-0x458]
       shr      eax, 31
       add      eax, dword ptr [rbp-0x458]
       sar      eax, 1
       imul     eax, dword ptr [rbp-0x74]
       jo       G_M000_IG103
       mov      dword ptr [rbp-0x78], eax
       xor      eax, eax
       mov      dword ptr [rbp-0x7C], eax
       jmp      G_M000_IG92
 
G_M000_IG85:                ;; offset=0x0D6D
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
       jne      SHORT G_M000_IG86
       mov      rdi, 0x76DAAF74AEF4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG97
 
G_M000_IG86:                ;; offset=0x0E04
       lea      rdi, [rbp+0x30]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xA8], rax
       mov      rax, bword ptr [rbp-0xA8]
       mov      qword ptr [rbp-0x438], rax
       mov      rax, qword ptr [rbp-0x438]
       mov      qword ptr [rbp-0x88], rax
       lea      rdi, [rbp-0x48]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xB0], rax
       mov      rax, bword ptr [rbp-0xB0]
       mov      qword ptr [rbp-0x440], rax
       mov      rax, qword ptr [rbp-0x440]
       mov      qword ptr [rbp-0x90], rax
       lea      rdi, [rbp+0x40]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xB8], rax
       mov      rax, bword ptr [rbp-0xB8]
       mov      qword ptr [rbp-0x448], rax
       mov      rax, qword ptr [rbp-0x448]
       mov      qword ptr [rbp-0x98], rax
       lea      rdi, [rbp+0x50]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0xC0], rax
       mov      rax, bword ptr [rbp-0xC0]
       mov      qword ptr [rbp-0x450], rax
       mov      rax, qword ptr [rbp-0x450]
       mov      qword ptr [rbp-0xA0], rax
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG87
       mov      rdi, qword ptr [rbp-0x88]
       mov      rsi, qword ptr [rbp-0x90]
       mov      rdx, qword ptr [rbp-0x98]
       mov      ecx, dword ptr [rbp+0x60]
       mov      r8d, dword ptr [rbp+0x68]
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int)]
       jmp      SHORT G_M000_IG88
 
G_M000_IG87:                ;; offset=0x0EE5
       mov      rdi, 0x76DAAF74AEF8
       call     CORINFO_HELP_COUNTPROFILE32
       mov      rdi, qword ptr [rbp-0x88]
       mov      rsi, qword ptr [rbp-0x90]
       mov      rdx, qword ptr [rbp-0x98]
       mov      ecx, dword ptr [rbp+0x60]
       mov      r8d, dword ptr [rbp+0x68]
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int)]
 
G_M000_IG88:                ;; offset=0x0F16
       mov      rdi, bword ptr [rbp+0x40]
       mov      rsi, qword ptr [rbp+0x48]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3E0], rax
       mov      qword ptr [rbp-0x3D8], rdx
       mov      rdi, bword ptr [rbp-0x3E0]
       mov      rsi, qword ptr [rbp-0x3D8]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG89
       mov      rdi, 0x76DAAF74AEFC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG97
 
G_M000_IG89:                ;; offset=0x0F5E
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG90
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
       jmp      SHORT G_M000_IG91
 
G_M000_IG90:                ;; offset=0x0F98
       mov      rdi, 0x76DAAF74AF00
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
 
G_M000_IG91:                ;; offset=0x0FD6
       mov      rdi, 0x76DAAF74AF04
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
 
G_M000_IG92:                ;; offset=0x1012
       mov      eax, dword ptr [rbp-0x420]
       dec      eax
       mov      dword ptr [rbp-0x420], eax
       cmp      dword ptr [rbp-0x420], 0
       jg       SHORT G_M000_IG94
 
G_M000_IG93:                ;; offset=0x1029
       lea      rdi, [rbp-0x420]
       mov      esi, 943
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG94:                ;; offset=0x103A
       mov      eax, dword ptr [rbp-0x7C]
       cmp      eax, dword ptr [rbp-0x78]
       jl       G_M000_IG85
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x3F0], rax
       mov      qword ptr [rbp-0x3E8], rdx
       mov      rdi, bword ptr [rbp-0x3F0]
       mov      rsi, qword ptr [rbp-0x3E8]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG96
       mov      rdi, bword ptr [rbp+0x50]
       mov      rsi, qword ptr [rbp+0x58]
       call     [System.Span`1[float]:op_Implicit(System.Span`1[float]):System.ReadOnlySpan`1[float]]
       mov      bword ptr [rbp-0x400], rax
       mov      qword ptr [rbp-0x3F8], rdx
       mov      rdi, bword ptr [rbp-0x400]
       mov      rsi, qword ptr [rbp-0x3F8]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG100
       mov      rdi, bword ptr [rbp-0x58]
       mov      rsi, qword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       SHORT G_M000_IG99
       mov      rdi, bword ptr [rbp+0x10]
       mov      rsi, qword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       jne      SHORT G_M000_IG101
 
G_M000_IG95:                ;; offset=0x10D2
       mov      rdi, 0x76DAAF74AF08
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG96:                ;; offset=0x10E1
       mov      rdi, 0x76DAAF74AF0C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG97:                ;; offset=0x10F0
       xor      eax, eax
 
G_M000_IG98:                ;; offset=0x10F2
       add      rsp, 0x490
       pop      rbp
       ret      
 
G_M000_IG99:                ;; offset=0x10FB
       mov      rdi, 0x76DAAF74AF10
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG96
 
G_M000_IG100:                ;; offset=0x110C
       mov      rdi, 0x76DAAF74AF14
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG96
 
G_M000_IG101:                ;; offset=0x111D
       mov      rdi, 0x76DAAF74AF18
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
 
G_M000_IG102:                ;; offset=0x11B1
       add      rsp, 0x490
       pop      rbp
       ret      
 
G_M000_IG103:                ;; offset=0x11BA
       call     CORINFO_HELP_OVERFLOW
       int3     
 
; Total bytes of code 4544

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 208
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
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       vbroadcastss ymm1, dword ptr [reloc @RWD04]
       lea      edx, [rsi-0x08]
       test     edx, edx
       jl       SHORT G_M000_IG05
       align    [9 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0040
       movsxd   r8, eax
       vpand    ymm2, ymm0, ymmword ptr [rcx+4*r8]
       vcmpgtps ymm2, ymm2, ymm1
       vmovmskps r8, ymm2
       test     r8d, r8d
       jne      SHORT G_M000_IG11
 
G_M000_IG04:                ;; offset=0x0057
       add      eax, 8
       cmp      eax, edx
       jle      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x005E
       xor      ecx, ecx
       mov      bword ptr [rbp-0x08], rcx
       cmp      eax, esi
       jl       SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x0068
       mov      eax, 1
 
G_M000_IG07:                ;; offset=0x006D
       vzeroupper 
       add      rsp, 16
       pop      rbp
       ret      
 
G_M000_IG08:                ;; offset=0x0076
       test     eax, eax
       jl       SHORT G_M000_IG13
 
G_M000_IG09:                ;; offset=0x007A
       mov      ecx, eax
       vmovss   xmm0, dword ptr [rdi+4*rcx]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       ja       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x0093
       inc      eax
       cmp      eax, esi
       jl       SHORT G_M000_IG09
       jmp      SHORT G_M000_IG06
 
G_M000_IG11:                ;; offset=0x009B
       xor      eax, eax
 
G_M000_IG12:                ;; offset=0x009D
       vzeroupper 
       add      rsp, 16
       pop      rbp
       ret      
 
G_M000_IG13:                ;; offset=0x00A6
       cmp      eax, esi
       jae      SHORT G_M000_IG15
       mov      ecx, eax
       vmovss   xmm0, dword ptr [rdi+4*rcx]
       vandps   xmm0, xmm0, xmmword ptr [reloc @RWD16]
       vucomiss xmm0, dword ptr [reloc @RWD04]
       ja       SHORT G_M000_IG11
 
G_M000_IG14:                ;; offset=0x00C3
       inc      eax
       cmp      eax, esi
       jl       SHORT G_M000_IG13
       jmp      SHORT G_M000_IG06
 
G_M000_IG15:                ;; offset=0x00CB
       call     CORINFO_HELP_RNGCHKFAIL
       int3     
 
RWD00  	dd	7FFFFFFFh		;       nan
RWD04  	dd	7E7FFFFFh		; 8.50706e+37
RWD08  	dd	00000000h, 00000000h
RWD16  	dq	7FFFFFFF7FFFFFFFh, 7FFFFFFF7FFFFFFFh

; Total bytes of code 209

; Assembly listing for method KernelAccess:PrepareWinograd(System.ReadOnlySpan`1[float],int,int,int):float[] (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       mov      rax, 0x76D2C9800190
       mov      rax, gword ptr [rax]
       mov      gword ptr [rbp-0x28], rax
       mov      rdi, gword ptr [rbp-0x28]
       mov      rsi, 0x76DAAF8E6D88
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
       mov      rdi, 0x76DAAF741E90
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
       mov      rdi, 0x76DAAF741E94
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
       mov      rdi, 0x76DAAF741E98
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
       mov      rdi, 0x76DAAF741E9C
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
       mov      rdi, 0x76DAAF741EA0
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
       mov      rdi, 0x76DAAF741EA4
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
       mov      rdi, 0x76DAAF741EA8
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
       mov      rdi, 0x76DAAF741EAC
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
       mov      rdi, 0x76DAAF741EB0
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
       mov      rdi, 0x76DAAF741EB4
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
       mov      rdi, 0x76DAAF741EB8
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
       mov      rdi, 0x76DAAF741EBC
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
       mov      rdi, 0x76DAAF741EC0
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
       mov      rdi, 0x76DAAF741EC4
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
       mov      rdi, 0x76DAAF741EC8
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
       mov      rdi, 0x76DAAF741ECC
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
       mov      rdi, 0x76DAAF741ED0
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
       mov      rdi, 0x76DAAF741ED4
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
       mov      rdi, 0x76DAAF741ED8
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
       mov      rdi, 0x76DAAF741EDC
       call     CORINFO_HELP_COUNTPROFILE32
       nop      
 
G_M000_IG54:                ;; offset=0x0DB5
       vzeroupper 
       add      rsp, 0x400
       pop      rbp
       ret      
 
; Total bytes of code 3521

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel256(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x1ec
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 1

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 64
       mov      qword ptr [rsp+0x448], r15
       mov      qword ptr [rsp+0x440], r14
       mov      qword ptr [rsp+0x438], r13
       mov      qword ptr [rsp+0x430], r12
       mov      qword ptr [rsp+0x428], rbx
       lea      rbp, [rsp+0x40]
       mov      rcx, qword ptr [rbp+0x3E0]
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      eax, dword ptr [rbp+0x428]
       mov      r9d, dword ptr [rbp+0x438]
       mov      r12d, dword ptr [rbp+0x3BC]
       mov      r13d, dword ptr [rbp+0x3A8]
       vmovups  ymm0, ymmword ptr [rbp+0x380]
       vmovups  ymm6, ymmword ptr [rbp+0x360]
       vmovups  ymm1, ymmword ptr [rbp+0x340]
       vmovups  ymm7, ymmword ptr [rbp+0x320]
       vmovups  ymm2, ymmword ptr [rbp+0x300]
       vmovups  ymm8, ymmword ptr [rbp+0x2E0]
       vmovups  ymm3, ymmword ptr [rbp+0x2C0]
       vmovups  ymm9, ymmword ptr [rbp+0x2A0]
       vmovups  ymm4, ymmword ptr [rbp+0x280]
       vmovups  ymm10, ymmword ptr [rbp+0x260]
       vmovups  ymm5, ymmword ptr [rbp+0x240]
       vmovups  ymm11, ymmword ptr [rbp+0x220]
       mov      rbx, qword ptr [rbp+0x218]
       mov      r15, qword ptr [rbp+0x210]
       mov      r10d, dword ptr [rbp+0x20C]
       mov      r14d, dword ptr [rbp+0x208]
       mov      r11d, dword ptr [rbp+0x204]
 
G_M000_IG02:                ;; offset=0x00EE
       jmp      G_M000_IG10
 
G_M000_IG03:                ;; offset=0x00F3
       inc      r10d
 
G_M000_IG04:                ;; offset=0x00F6
       cmp      r10d, edx
       jge      G_M000_IG12
 
G_M000_IG05:                ;; offset=0x00FF
       xor      r11d, r11d
       mov      r14d, r11d
       jmp      SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x0107
       inc      r14d
       cmp      r14d, 3
       jge      SHORT G_M000_IG03
 
G_M000_IG07:                ;; offset=0x0110
       xor      r11d, r11d
       mov      dword ptr [rbp+0x3CC], edx
       mov      qword ptr [rbp+0x3D0], rdi
       mov      dword ptr [rbp+0x3C8], esi
       mov      dword ptr [rbp+0x438], r9d
 
G_M000_IG08:                ;; offset=0x012D
       vmovups  ymm12, ymmword ptr [rbx]
       vmovups  ymm13, ymmword ptr [r15]
       mov      r8d, r10d
       sar      r8d, 31
       and      r8d, 7
       add      r8d, r10d
       sar      r8d, 3
       mov      r9d, dword ptr [rbp+0x3C0]
       imul     r8d, r9d
       mov      esi, dword ptr [rbp+0x3AC]
       mov      edi, esi
       imul     edi, eax
       add      edi, r8d
       add      edi, r14d
       imul     edi, r12d
       mov      r8d, r13d
       imul     r8d, eax
       add      edi, r8d
       add      edi, r11d
       shl      edi, 3
       movsxd   rdi, edi
       shl      rdi, 2
       add      rdi, rcx
       mov      r8d, r10d
       sar      r8d, 31
       and      r8d, 7
       add      r8d, r10d
       and      r8d, -8
       mov      edx, r10d
       sub      edx, r8d
       movsxd   rdx, edx
       lea      rdx, [rdi+4*rdx]
       vbroadcastss ymm14, dword ptr [rdx]
       vfmadd231ps ymm0, ymm12, ymm14
       vfmadd231ps ymm6, ymm13, ymm14
       lea      edi, [8*rax]
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm1, ymm12, ymm14
       vfmadd231ps ymm7, ymm13, ymm14
       lea      edi, [rax+rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm2, ymm12, ymm14
       vfmadd231ps ymm8, ymm13, ymm14
       lea      edi, [rax+2*rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm3, ymm12, ymm14
       vfmadd231ps ymm9, ymm13, ymm14
       lea      edi, [4*rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm4, ymm12, ymm14
       vfmadd231ps ymm10, ymm13, ymm14
       lea      edi, [rax+4*rax]
       shl      edi, 3
       movsxd   rdi, edi
       vbroadcastss ymm14, dword ptr [rdx+4*rdi]
       vfmadd231ps ymm5, ymm12, ymm14
       vfmadd231ps ymm11, ymm13, ymm14
       add      rbx, 32
       add      r15, 32
       inc      r11d
       mov      dword ptr [rbp+0x3AC], esi
       mov      dword ptr [rbp+0x3C0], r9d
 
G_M000_IG09:                ;; offset=0x024A
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r9d, dword ptr [rbp+0x438]
 
G_M000_IG10:                ;; offset=0x0264
       cmp      r11d, 3
       jge      G_M000_IG06
 
G_M000_IG11:                ;; offset=0x026E
       mov      dword ptr [rbp+0x3CC], edx
       mov      qword ptr [rbp+0x3D0], rdi
       mov      dword ptr [rbp+0x3C8], esi
       mov      dword ptr [rbp+0x438], r9d
       jmp      G_M000_IG08
 
G_M000_IG12:                ;; offset=0x028D
       mov      r10d, dword ptr [rbp+0x3B0]
       mov      r11d, r10d
       sar      r11d, 31
       and      r11d, 7
       mov      dword ptr [rbp+0x3B0], r10d
       add      r11d, r10d
       sar      r11d, 3
       mov      r14d, dword ptr [rbp+0x3B8]
       mov      ebx, r11d
       imul     ebx, r14d
       mov      r15d, dword ptr [rbp+0x3AC]
       mov      dword ptr [rbp+0x438], r9d
       mov      r8d, r15d
       imul     r8d, r9d
       add      ebx, r8d
       add      ebx, r13d
       shl      ebx, 3
       movsxd   r10, ebx
       vmovups  ymmword ptr [rdi+4*r10], ymm0
       mov      r10d, dword ptr [rbp+0x3B0]
       add      r10d, 8
       cmp      r10d, esi
       jge      SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x02F2
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       shl      r9d, 3
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm6
 
G_M000_IG14:                ;; offset=0x030D
       lea      r9d, [rbx+0x08]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm1
       cmp      r10d, esi
       jge      SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x031F
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       lea      r9d, [8*r9+0x08]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm7
 
G_M000_IG16:                ;; offset=0x033E
       lea      r9d, [rbx+0x10]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm2
       cmp      r10d, esi
       jge      SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x0350
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       lea      r9d, [8*r9+0x10]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm8
 
G_M000_IG18:                ;; offset=0x036F
       lea      r9d, [rbx+0x18]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm3
       cmp      r10d, esi
       jge      SHORT G_M000_IG20
 
G_M000_IG19:                ;; offset=0x0381
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       lea      r9d, [8*r9+0x18]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm9
 
G_M000_IG20:                ;; offset=0x03A0
       lea      r9d, [rbx+0x20]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm4
       cmp      r10d, esi
       jge      SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x03B2
       lea      r9d, [r11+0x01]
       imul     r9d, r14d
       add      r9d, r8d
       add      r9d, r13d
       lea      r9d, [8*r9+0x20]
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm10
 
G_M000_IG22:                ;; offset=0x03D1
       add      ebx, 40
       movsxd   r9, ebx
       vmovups  ymmword ptr [rdi+4*r9], ymm5
       mov      dword ptr [rbp+0x3C8], esi
       cmp      r10d, esi
       jge      SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x03E8
       inc      r11d
       mov      dword ptr [rbp+0x3B8], r14d
       imul     r11d, r14d
       add      r8d, r11d
       add      r8d, r13d
       lea      r8d, [8*r8+0x28]
       movsxd   r8, r8d
       mov      qword ptr [rbp+0x3D0], rdi
       vmovups  ymmword ptr [rdi+4*r8], ymm11
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r14d, dword ptr [rbp+0x3B8]
 
G_M000_IG24:                ;; offset=0x0422
       add      r13d, 6
 
G_M000_IG25:                ;; offset=0x0426
       lea      r8d, [r13+0x06]
       mov      r9d, dword ptr [rbp+0x438]
       cmp      r8d, r9d
       jg       G_M000_IG33
 
G_M000_IG26:                ;; offset=0x043A
       mov      r8d, r15d
       imul     r8d, r9d
       lea      r8d, [r8+r13+0x06]
       cmp      r8d, dword ptr [rbp+0x3B4]
       jg       G_M000_IG33
 
G_M000_IG27:                ;; offset=0x0453
       mov      dword ptr [rbp+0x438], r9d
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
       mov      ebx, dword ptr [rbp+0x3B0]
       mov      r11d, ebx
       imul     r11d, edx
       lea      r11d, [r11+8*r11]
       movsxd   r11, r11d
       mov      r10, qword ptr [rbp+0x3D8]
       lea      r11, [r10+4*r11]
       mov      dword ptr [rbp+0x3CC], edx
       lea      r10d, [8*rdx]
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r11+4*r10]
       xor      r9d, r9d
       mov      dword ptr [rbp+0x3AC], r15d
       mov      dword ptr [rbp+0x3B8], r14d
       mov      dword ptr [rbp+0x3B0], ebx
       mov      rbx, r11
       mov      r15, r10
       mov      r10d, r9d
       mov      edx, dword ptr [rbp+0x3CC]
       mov      esi, dword ptr [rbp+0x3C8]
       mov      r9d, dword ptr [rbp+0x438]
       jmp      G_M000_IG04
       align    [0 bytes for IG28]
 
G_M000_IG28:                ;; offset=0x04FE
       vmulps   ymm7, ymm1, ymm7
       vaddps   ymm0, ymm7, ymm0
       vmulps   ymm1, ymm1, ymm2
       vaddps   ymm6, ymm1, ymm6
       jmp      G_M000_IG43
 
G_M000_IG29:                ;; offset=0x0513
       vmulps   ymm8, ymm1, ymm7
       vaddps   ymm0, ymm8, ymm0
       vmulps   ymm3, ymm1, ymm2
       vaddps   ymm6, ymm3, ymm6
       jmp      G_M000_IG45
 
G_M000_IG30:                ;; offset=0x0528
       mov      r10d, dword ptr [rbp+0x3B0]
       mov      esi, r10d
       sar      esi, 31
       and      esi, 7
       add      esi, r10d
       sar      esi, 3
       mov      r14d, dword ptr [rbp+0x3B8]
       mov      r9d, esi
       imul     r9d, r14d
       mov      r11d, dword ptr [rbp+0x438]
       mov      ebx, r15d
       imul     ebx, r11d
       add      r9d, ebx
       add      r9d, r13d
       shl      r9d, 3
       movsxd   r9, r9d
       vmovups  ymmword ptr [rdi+4*r9], ymm0
       lea      r9d, [r10+0x08]
       mov      ebx, dword ptr [rbp+0x3C8]
       cmp      r9d, ebx
       jge      SHORT G_M000_IG32
 
G_M000_IG31:                ;; offset=0x057C
       inc      esi
       imul     esi, r14d
       mov      dword ptr [rbp+0x438], r11d
       mov      r9d, r15d
       imul     r9d, r11d
       add      esi, r9d
       add      esi, r13d
       shl      esi, 3
       movsxd   rsi, esi
       mov      qword ptr [rbp+0x3D0], rdi
       vmovups  ymmword ptr [rdi+4*rsi], ymm6
       mov      rdi, qword ptr [rbp+0x3D0]
       mov      r11d, dword ptr [rbp+0x438]
 
G_M000_IG32:                ;; offset=0x05B6
       inc      r13d
       mov      dword ptr [rbp+0x3C0], r8d
       mov      dword ptr [rbp+0x3C8], ebx
       mov      dword ptr [rbp+0x3B0], r10d
       mov      r9d, r11d
 
G_M000_IG33:                ;; offset=0x05D0
       cmp      r13d, r9d
       jge      G_M000_IG35
 
G_M000_IG34:                ;; offset=0x05D9
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       mov      r11d, edx
       imul     r11d, dword ptr [rbp+0x3B0]
       lea      r11d, [r11+8*r11]
       movsxd   r11, r11d
       mov      rbx, qword ptr [rbp+0x3D8]
       lea      r11, [rbx+4*r11]
       lea      r10d, [8*rdx]
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r11+4*r10]
       mov      dword ptr [rbp+0x438], r9d
       mov      ebx, r15d
       imul     ebx, r9d
       add      ebx, r13d
       cmp      ebx, dword ptr [rbp+0x3B4]
       setl     bl
       movzx    rbx, bl
       mov      dword ptr [rbp+0x16C], ebx
       xor      r9d, r9d
       mov      r8d, r15d
       imul     r8d, eax
       mov      dword ptr [rbp-0x30], r8d
       mov      r8d, r13d
       imul     r8d, eax
       mov      dword ptr [rbp-0x34], r8d
       cmp      r9d, edx
       mov      dword ptr [rbp+0x3B8], r14d
       jl       SHORT G_M000_IG39
       jmp      G_M000_IG50
 
G_M000_IG35:                ;; offset=0x065E
       inc      r15d
       mov      r11d, dword ptr [rbp+0x430]
       cmp      r15d, r11d
       jge      G_M000_IG51
 
G_M000_IG36:                ;; offset=0x0671
       xor      r13d, r13d
       mov      dword ptr [rbp+0x438], r9d
       mov      dword ptr [rbp+0x430], r11d
       jmp      G_M000_IG25
 
G_M000_IG37:                ;; offset=0x0687
       mov      rdi, qword ptr [rbp+0x3D0]
       inc      r9d
       cmp      r9d, edx
       jge      G_M000_IG30
 
G_M000_IG38:                ;; offset=0x069A
       mov      dword ptr [rbp+0x3C0], r8d
 
G_M000_IG39:                ;; offset=0x06A1
       xor      r14d, r14d
       mov      esi, r9d
       sar      esi, 31
       and      esi, 7
       add      esi, r9d
       sar      esi, 3
       mov      r8d, dword ptr [rbp+0x3C0]
       imul     esi, r8d
       add      esi, dword ptr [rbp-0x30]
       mov      dword ptr [rbp-0x2C], esi
       mov      qword ptr [rbp+0x3D0], rdi
       jmp      G_M000_IG46
 
G_M000_IG40:                ;; offset=0x06D0
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
 
G_M000_IG41:                ;; offset=0x06DA
       add      r11, 32
       add      r10, 32
       lea      edi, [rsi+0x01]
       lea      edi, [rbx+8*rdi]
       movsxd   rdi, edi
       vbroadcastss ymm1, dword ptr [rcx+4*rdi]
       vmovups  ymm7, ymmword ptr [r11]
       vmovups  ymm2, ymmword ptr [r10]
       mov      edi, dword ptr [rbp+0x16C]
       test     edi, edi
       je       G_M000_IG28
 
G_M000_IG42:                ;; offset=0x0709
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
 
G_M000_IG43:                ;; offset=0x0713
       add      r11, 32
       add      r10, 32
       add      esi, 2
       lea      esi, [rbx+8*rsi]
       movsxd   rsi, esi
       vbroadcastss ymm1, dword ptr [rcx+4*rsi]
       vmovups  ymm7, ymmword ptr [r11]
       vmovups  ymm2, ymmword ptr [r10]
       test     edi, edi
       je       G_M000_IG29
 
G_M000_IG44:                ;; offset=0x073C
       vfmadd231ps ymm0, ymm7, ymm1
       vfmadd231ps ymm6, ymm2, ymm1
 
G_M000_IG45:                ;; offset=0x0746
       add      r11, 32
       add      r10, 32
       inc      r14d
       cmp      r14d, 3
       mov      esi, dword ptr [rbp-0x2C]
       jge      G_M000_IG37
 
G_M000_IG46:                ;; offset=0x075E
       add      esi, r14d
       imul     esi, r12d
       add      esi, dword ptr [rbp-0x34]
       mov      edi, r9d
       sar      edi, 31
       and      edi, 7
       add      edi, r9d
       and      edi, -8
       mov      ebx, r9d
       sub      ebx, edi
       lea      edi, [rbx+8*rsi]
       movsxd   rdi, edi
       vbroadcastss ymm1, dword ptr [rcx+4*rdi]
       vmovups  ymm7, ymmword ptr [r11]
       vmovups  ymm2, ymmword ptr [r10]
       cmp      dword ptr [rbp+0x16C], 0
       jne      G_M000_IG40
 
G_M000_IG47:                ;; offset=0x079F
       vmulps   ymm7, ymm1, ymm7
       vaddps   ymm0, ymm7, ymm0
       vmulps   ymm1, ymm1, ymm2
       vaddps   ymm6, ymm1, ymm6
       jmp      G_M000_IG41
 
G_M000_IG48:                ;; offset=0x07B4
       xor      r13d, r13d
       mov      dword ptr [rbp+0x430], r11d
       test     r11d, r11d
       mov      dword ptr [rbp+0x3B0], r15d
       mov      r11d, dword ptr [rbp+0x430]
       jle      SHORT G_M000_IG51
 
G_M000_IG49:                ;; offset=0x07D1
       mov      r15d, r13d
       jmp      G_M000_IG36
 
G_M000_IG50:                ;; offset=0x07D9
       mov      r8d, dword ptr [rbp+0x3C0]
       jmp      G_M000_IG30
 
G_M000_IG51:                ;; offset=0x07E5
       mov      r15d, dword ptr [rbp+0x3B0]
       add      r15d, 16
       cmp      r15d, dword ptr [rbp+0x3C8]
       jl       SHORT G_M000_IG48
 
G_M000_IG52:                ;; offset=0x07F9
       vzeroupper 
       add      rsp, 0x428
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 2062

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Dynamic PGO
; rbp based frame
; fully interruptible
; with Dynamic PGO: fgCalledCount is 88

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

; Assembly listing for method KernelAccess:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       mov      rsi, 0x76D2C9800198
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
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
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
       mov      r9, 0x76D2C98001A8
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
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 96
; 44 inlinees with PGO data; 189 single block inlinees; 1 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 408
       lea      rbp, [rsp+0x1C0]
       xor      eax, eax
       mov      qword ptr [rbp-0x158], rax
       vxorps   xmm8, xmm8, xmm8
       mov      rax, -288
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x20], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x10], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x30], rax
       mov      bword ptr [rbp-0x178], rdi
       mov      dword ptr [rbp-0x15C], esi
       mov      bword ptr [rbp-0x180], rdx
       mov      dword ptr [rbp-0x160], ecx
       mov      bword ptr [rbp-0x188], r8
       mov      dword ptr [rbp-0x164], r9d
       mov      r15d, dword ptr [rbp+0x60]
       mov      ebx, dword ptr [rbp+0x68]
       mov      eax, dword ptr [rbp+0x70]
       mov      esi, dword ptr [rbp+0x78]
       mov      r12, bword ptr [rbp+0x30]
       mov      r14d, dword ptr [rbp+0x38]
       mov      r13d, dword ptr [rbp+0x48]
 
G_M000_IG02:                ;; offset=0x008E
       cmp      dword ptr [rbp+0x80], 8
       jne      G_M000_IG95
 
G_M000_IG03:                ;; offset=0x009B
       cmp      r15d, 16
       jl       G_M000_IG96
       test     r15b, 15
       jne      G_M000_IG96
       cmp      ebx, 32
       jl       G_M000_IG96
       test     bl, 15
       jne      G_M000_IG96
       test     eax, eax
       jle      G_M000_IG96
       test     esi, esi
       setle    r8b
       movzx    r8, r8b
 
G_M000_IG04:                ;; offset=0x00D3
       movzx    r8, r8b
       test     r8d, r8d
       jne      G_M000_IG97
       mov      r8d, eax
       add      r8d, 2
       jo       G_M000_IG121
       mov      dword ptr [rbp-0x18C], r8d
       mov      r8d, esi
       add      r8d, 2
       jo       G_M000_IG121
       imul     r8d, dword ptr [rbp-0x18C]
       jo       G_M000_IG121
       imul     r8d, r15d
       jo       G_M000_IG121
       mov      dword ptr [rbp+0x70], eax
       mov      r8d, eax
       add      r8d, 1
       jo       G_M000_IG121
       mov      dword ptr [rbp-0x170], r8d
       sub      r8d, 1
       jo       G_M000_IG121
       imul     r8d, ebx
       jo       G_M000_IG121
       mov      dword ptr [rbp-0x18C], r8d
       mov      dword ptr [rbp+0x78], esi
       mov      r8d, esi
       add      r8d, 1
       jo       G_M000_IG121
       sub      r8d, 1
       jo       G_M000_IG121
       imul     r8d, dword ptr [rbp-0x18C]
       jo       G_M000_IG121
       imul     r8d, dword ptr [rbp+0x80], 2
       jo       G_M000_IG121
       mov      dword ptr [rbp-0x16C], r8d
       add      r8d, ebx
       jo       G_M000_IG121
       mov      eax, r8d
       sub      eax, 1
       jo       G_M000_IG121
       mov      r8d, dword ptr [rbp-0x16C]
       cdq      
       idiv     edx:eax, r8d
       imul     r8d, eax
       jo       G_M000_IG121
 
G_M000_IG05:                ;; offset=0x01B2
       imul     r8d, r15d
       jo       G_M000_IG121
       imul     r8d, r8d, 9
       jo       G_M000_IG121
       lea      r8, [rbp-0x40]
       mov      qword ptr [rsp], r8
       lea      r8, [rbp-0x30]
       lea      r9, [rbp-0x38]
       mov      edi, r15d
       mov      esi, ebx
       mov      edx, dword ptr [rbp+0x70]
       mov      ecx, dword ptr [rbp+0x78]
       call     [Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool]
       test     eax, eax
       je       G_M000_IG119
       mov      edi, r15d
       imul     edi, dword ptr [rbp+0x70]
       jo       G_M000_IG121
       imul     edi, dword ptr [rbp+0x78]
       jo       G_M000_IG121
       mov      edx, dword ptr [rbp-0x15C]
       cmp      edi, edx
       jne      G_M000_IG98
       imul     edi, r15d, 16
       jo       G_M000_IG121
       imul     edi, ebx
       jo       G_M000_IG121
       mov      esi, dword ptr [rbp-0x160]
       cmp      edi, esi
       jne      G_M000_IG98
       mov      r8d, dword ptr [rbp+0x28]
       cmp      r8d, dword ptr [rbp-0x40]
       jne      G_M000_IG98
       mov      r9d, dword ptr [rbp-0x164]
       test     r9d, r9d
       je       SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x024F
       mov      dword ptr [rbp-0x164], r9d
       cmp      r9d, ebx
       mov      r9d, dword ptr [rbp-0x164]
       jne      G_M000_IG98
 
G_M000_IG07:                ;; offset=0x0266
       mov      r10d, dword ptr [rbp+0x18]
       test     r10d, r10d
       je       SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x026F
       cmp      r10d, dword ptr [rbp-0x40]
       jne      G_M000_IG98
 
G_M000_IG09:                ;; offset=0x0279
       cmp      r14d, dword ptr [rbp-0x30]
       jl       G_M000_IG98
       cmp      r13d, dword ptr [rbp-0x38]
       jl       G_M000_IG98
       mov      r11d, dword ptr [rbp+0x58]
       cmp      r11d, dword ptr [rbp-0x40]
       jl       G_M000_IG98
       mov      edi, dword ptr [rbp-0x30]
       cmp      edi, r14d
       ja       G_M000_IG99
       mov      r14d, edi
       mov      edi, dword ptr [rbp-0x38]
       cmp      edi, r13d
       ja       G_M000_IG99
       mov      r13, bword ptr [rbp+0x40]
       mov      r8d, dword ptr [rbp-0x40]
       cmp      r8d, r11d
       mov      dword ptr [rbp-0x164], r9d
       ja       G_M000_IG99
       mov      r11, bword ptr [rbp+0x50]
       mov      r9d, edx
       test     r9d, r9d
       je       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x02DA
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG11
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp-0x178]
       mov      qword ptr [rbp-0x80], r9
       mov      r9d, edx
       shl      r9, 2
       cmp      qword ptr [rbp-0x80], r9
       jb       G_M000_IG100
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x80]
       jb       G_M000_IG100
 
G_M000_IG11:                ;; offset=0x0315
       mov      r9d, esi
       test     r9d, r9d
       je       SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x031D
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG13
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp-0x180]
       mov      qword ptr [rbp-0x88], r9
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0x88], r9
       jb       G_M000_IG100
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x88]
       jb       G_M000_IG100
 
G_M000_IG13:                ;; offset=0x0361
       mov      r9d, dword ptr [rbp-0x164]
       test     r9d, r9d
       je       SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x036D
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG15
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp-0x188]
       mov      qword ptr [rbp-0x90], r9
       mov      r9d, dword ptr [rbp-0x164]
       shl      r9, 2
       cmp      qword ptr [rbp-0x90], r9
       jb       G_M000_IG100
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x90]
       jb       G_M000_IG100
 
G_M000_IG15:                ;; offset=0x03B5
       mov      r9d, r10d
       test     r9d, r9d
       je       SHORT G_M000_IG17
 
G_M000_IG16:                ;; offset=0x03BD
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG17
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0x98], r9
       mov      r9d, r10d
       shl      r9, 2
       cmp      qword ptr [rbp-0x98], r9
       jb       G_M000_IG100
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x98]
       jb       G_M000_IG100
 
G_M000_IG17:                ;; offset=0x03FE
       mov      r9d, edx
       test     r9d, r9d
       je       SHORT G_M000_IG19
 
G_M000_IG18:                ;; offset=0x0406
       test     r14d, r14d
       je       SHORT G_M000_IG19
       mov      rax, r12
       sub      rax, qword ptr [rbp-0x178]
       mov      qword ptr [rbp-0xA0], rax
       mov      eax, edx
       shl      rax, 2
       cmp      qword ptr [rbp-0xA0], rax
       jb       G_M000_IG100
       mov      eax, r14d
       shl      rax, 2
       neg      rax
       cmp      rax, qword ptr [rbp-0xA0]
       jb       G_M000_IG100
 
G_M000_IG19:                ;; offset=0x0446
       mov      r9d, esi
       test     r9d, r9d
       je       SHORT G_M000_IG21
 
G_M000_IG20:                ;; offset=0x044E
       test     r14d, r14d
       je       SHORT G_M000_IG21
       mov      rax, r12
       sub      rax, qword ptr [rbp-0x180]
       mov      qword ptr [rbp-0xA8], rax
       mov      eax, esi
       shl      rax, 2
       cmp      qword ptr [rbp-0xA8], rax
       jb       G_M000_IG100
       mov      eax, r14d
       shl      rax, 2
       neg      rax
       cmp      rax, qword ptr [rbp-0xA8]
       jb       G_M000_IG100
 
G_M000_IG21:                ;; offset=0x048E
       mov      r9d, dword ptr [rbp-0x164]
       test     r9d, r9d
       je       SHORT G_M000_IG23
 
G_M000_IG22:                ;; offset=0x049A
       test     r14d, r14d
       je       SHORT G_M000_IG23
       mov      rax, r12
       sub      rax, qword ptr [rbp-0x188]
       mov      qword ptr [rbp-0xB0], rax
       mov      eax, dword ptr [rbp-0x164]
       mov      r9d, eax
       shl      r9, 2
       cmp      qword ptr [rbp-0xB0], r9
       jb       G_M000_IG100
       mov      r9d, r14d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xB0]
       mov      dword ptr [rbp-0x164], eax
       jb       G_M000_IG100
 
G_M000_IG23:                ;; offset=0x04E7
       mov      r9d, r10d
       test     r9d, r9d
       je       SHORT G_M000_IG25
 
G_M000_IG24:                ;; offset=0x04EF
       test     r14d, r14d
       je       SHORT G_M000_IG25
       mov      rax, r12
       sub      rax, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xB8], rax
       mov      eax, r10d
       shl      rax, 2
       cmp      qword ptr [rbp-0xB8], rax
       jb       G_M000_IG100
       mov      eax, r14d
       shl      rax, 2
       neg      rax
       cmp      rax, qword ptr [rbp-0xB8]
       jb       G_M000_IG100
 
G_M000_IG25:                ;; offset=0x052D
       mov      r9d, edx
       test     r9d, r9d
       je       SHORT G_M000_IG27
 
G_M000_IG26:                ;; offset=0x0535
       test     edi, edi
       je       SHORT G_M000_IG27
       mov      rax, r13
       sub      rax, qword ptr [rbp-0x178]
       mov      qword ptr [rbp-0xC0], rax
       mov      eax, edx
       shl      rax, 2
       cmp      qword ptr [rbp-0xC0], rax
       jb       G_M000_IG100
       mov      eax, edi
       shl      rax, 2
       neg      rax
       cmp      rax, qword ptr [rbp-0xC0]
       jb       G_M000_IG100
 
G_M000_IG27:                ;; offset=0x0573
       mov      r9d, esi
       test     r9d, r9d
       je       SHORT G_M000_IG29
 
G_M000_IG28:                ;; offset=0x057B
       test     edi, edi
       je       SHORT G_M000_IG29
       mov      rax, r13
       sub      rax, qword ptr [rbp-0x180]
       mov      qword ptr [rbp-0xC8], rax
       mov      eax, esi
       shl      rax, 2
       cmp      qword ptr [rbp-0xC8], rax
       jb       G_M000_IG100
       mov      eax, edi
       shl      rax, 2
       neg      rax
       cmp      rax, qword ptr [rbp-0xC8]
       jb       G_M000_IG100
 
G_M000_IG29:                ;; offset=0x05B9
       mov      r9d, dword ptr [rbp-0x164]
       test     r9d, r9d
       je       SHORT G_M000_IG31
 
G_M000_IG30:                ;; offset=0x05C5
       test     edi, edi
       je       SHORT G_M000_IG31
       mov      rax, r13
       sub      rax, qword ptr [rbp-0x188]
       mov      qword ptr [rbp-0xD0], rax
       mov      eax, dword ptr [rbp-0x164]
       mov      r9d, eax
       shl      r9, 2
       cmp      qword ptr [rbp-0xD0], r9
       jb       G_M000_IG100
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xD0]
       mov      dword ptr [rbp-0x164], eax
       jb       G_M000_IG100
 
G_M000_IG31:                ;; offset=0x0611
       mov      r9d, r10d
       test     r9d, r9d
       je       SHORT G_M000_IG33
 
G_M000_IG32:                ;; offset=0x0619
       test     edi, edi
       je       SHORT G_M000_IG33
       mov      rax, r13
       sub      rax, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xD8], rax
       mov      eax, r10d
       shl      rax, 2
       cmp      qword ptr [rbp-0xD8], rax
       jb       G_M000_IG100
       mov      eax, edi
       shl      rax, 2
       neg      rax
       cmp      rax, qword ptr [rbp-0xD8]
       jb       G_M000_IG100
 
G_M000_IG33:                ;; offset=0x0655
       mov      r9d, edx
       test     r9d, r9d
       je       SHORT G_M000_IG35
 
G_M000_IG34:                ;; offset=0x065D
       test     r8d, r8d
       je       SHORT G_M000_IG35
       mov      rax, r11
       sub      rax, qword ptr [rbp-0x178]
       mov      qword ptr [rbp-0xE0], rax
       mov      eax, edx
       shl      rax, 2
       cmp      qword ptr [rbp-0xE0], rax
       jb       G_M000_IG100
       mov      eax, r8d
       shl      rax, 2
       neg      rax
       cmp      rax, qword ptr [rbp-0xE0]
       jb       G_M000_IG100
 
G_M000_IG35:                ;; offset=0x069D
       mov      r9d, esi
       test     r9d, r9d
       je       SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x06A5
       test     r8d, r8d
       je       SHORT G_M000_IG37
       mov      rax, r11
       sub      rax, qword ptr [rbp-0x180]
       mov      qword ptr [rbp-0xE8], rax
       mov      eax, esi
       shl      rax, 2
       cmp      qword ptr [rbp-0xE8], rax
       jb       G_M000_IG100
       mov      eax, r8d
       shl      rax, 2
       neg      rax
       cmp      rax, qword ptr [rbp-0xE8]
       jb       G_M000_IG100
 
G_M000_IG37:                ;; offset=0x06E5
       mov      r9d, dword ptr [rbp-0x164]
       test     r9d, r9d
       je       SHORT G_M000_IG39
 
G_M000_IG38:                ;; offset=0x06F1
       test     r8d, r8d
       je       SHORT G_M000_IG39
       mov      rax, r11
       sub      rax, qword ptr [rbp-0x188]
       mov      qword ptr [rbp-0xF0], rax
       mov      eax, dword ptr [rbp-0x164]
       mov      r9d, eax
       shl      r9, 2
       cmp      qword ptr [rbp-0xF0], r9
       jb       G_M000_IG100
       mov      r9d, r8d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xF0]
       mov      dword ptr [rbp-0x164], eax
       jb       G_M000_IG100
 
G_M000_IG39:                ;; offset=0x073E
       mov      r9d, r10d
       test     r9d, r9d
       je       SHORT G_M000_IG41
 
G_M000_IG40:                ;; offset=0x0746
       test     r8d, r8d
       je       SHORT G_M000_IG41
       mov      rax, r11
       sub      rax, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xF8], rax
       mov      eax, r10d
       shl      rax, 2
       cmp      qword ptr [rbp-0xF8], rax
       jb       G_M000_IG100
       mov      eax, r8d
       shl      rax, 2
       neg      rax
       cmp      rax, qword ptr [rbp-0xF8]
       jb       G_M000_IG100
 
G_M000_IG41:                ;; offset=0x0784
       mov      r9d, r14d
       test     r9d, r9d
       je       SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x078C
       test     edi, edi
       je       SHORT G_M000_IG43
       mov      r9, r13
       sub      r9, r12
       mov      qword ptr [rbp-0x100], r9
       mov      r9d, r14d
       shl      r9, 2
       cmp      qword ptr [rbp-0x100], r9
       jb       G_M000_IG100
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x100]
       jb       G_M000_IG100
 
G_M000_IG43:                ;; offset=0x07C8
       mov      r9d, r14d
       test     r9d, r9d
       je       SHORT G_M000_IG45
 
G_M000_IG44:                ;; offset=0x07D0
       test     r8d, r8d
       je       SHORT G_M000_IG45
       mov      r9, r11
       sub      r9, r12
       mov      qword ptr [rbp-0x108], r9
       mov      r9d, r14d
       shl      r9, 2
       cmp      qword ptr [rbp-0x108], r9
       jb       G_M000_IG100
       mov      r9d, r8d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x108]
       jb       G_M000_IG100
 
G_M000_IG45:                ;; offset=0x080D
       mov      r9d, r14d
       test     r9d, r9d
       je       SHORT G_M000_IG47
 
G_M000_IG46:                ;; offset=0x0815
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG47
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, r12
       mov      qword ptr [rbp-0x110], r9
       mov      r9d, r14d
       shl      r9, 2
       cmp      qword ptr [rbp-0x110], r9
       jb       G_M000_IG100
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x110]
       jb       G_M000_IG100
 
G_M000_IG47:                ;; offset=0x0855
       mov      r9d, edi
       test     r9d, r9d
       je       SHORT G_M000_IG49
 
G_M000_IG48:                ;; offset=0x085D
       test     r8d, r8d
       je       SHORT G_M000_IG49
       mov      r9, r11
       sub      r9, r13
       mov      qword ptr [rbp-0x118], r9
       mov      r9d, edi
       shl      r9, 2
       cmp      qword ptr [rbp-0x118], r9
       jb       G_M000_IG100
       mov      r9d, r8d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x118]
       jb       G_M000_IG100
 
G_M000_IG49:                ;; offset=0x089A
       mov      r9d, edi
       test     r9d, r9d
       je       SHORT G_M000_IG51
 
G_M000_IG50:                ;; offset=0x08A2
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG51
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, r13
       mov      qword ptr [rbp-0x120], r9
       mov      dword ptr [rbp+0x48], edi
       mov      r9d, edi
       shl      r9, 2
       cmp      qword ptr [rbp-0x120], r9
       jb       G_M000_IG100
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x120]
       mov      edi, dword ptr [rbp+0x48]
       jb       G_M000_IG100
 
G_M000_IG51:                ;; offset=0x08E8
       mov      r9d, r8d
       test     r9d, r9d
       je       SHORT G_M000_IG53
 
G_M000_IG52:                ;; offset=0x08F0
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG53
       mov      r9, bword ptr [rbp+0x20]
       mov      bword ptr [rbp+0x50], r11
       sub      r9, r11
       mov      qword ptr [rbp-0x128], r9
       mov      dword ptr [rbp+0x58], r8d
       mov      r9d, r8d
       shl      r9, 2
       cmp      qword ptr [rbp-0x128], r9
       jb       G_M000_IG100
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x128]
       mov      r8d, dword ptr [rbp+0x58]
       mov      r11, bword ptr [rbp+0x50]
       jb       G_M000_IG100
 
G_M000_IG53:                ;; offset=0x0940
       cmp      dword ptr [rbp+0x80], 16
       je       G_M000_IG101
       mov      dword ptr [rbp-0x168], edx
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r9, r9
       cmp      dword ptr [rbp-0x168], 0
       cmovne   r9, bword ptr [rbp-0x178]
       mov      bword ptr [rbp-0x130], r9
       xor      eax, eax
       cmp      edx, 8
       jl       SHORT G_M000_IG55
       align    [4 bytes for IG54]
 
G_M000_IG54:                ;; offset=0x0980
       mov      ecx, eax
       sar      ecx, 31
       and      ecx, 7
       add      ecx, eax
       sar      ecx, 3
       movsxd   rcx, ecx
       shl      rcx, 5
       vpand    ymm1, ymm0, ymmword ptr [rcx+r9]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG102
       add      eax, 8
       lea      ecx, [rax+0x08]
       cmp      ecx, edx
       jle      SHORT G_M000_IG54
 
G_M000_IG55:                ;; offset=0x09B3
       cmp      eax, edx
       jl       G_M000_IG103
       xor      ecx, ecx
       mov      bword ptr [rbp-0x130], rcx
       mov      ecx, 1
 
G_M000_IG56:                ;; offset=0x09C9
       xor      r9, r9
       mov      bword ptr [rbp-0x130], r9
       test     ecx, ecx
       je       G_M000_IG119
       mov      ecx, esi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r9, r9
       test     ecx, ecx
       cmovne   r9, bword ptr [rbp-0x180]
       mov      bword ptr [rbp-0x138], r9
       mov      rcx, r9
       xor      r9d, r9d
       cmp      esi, 8
       jl       SHORT G_M000_IG58
       align    [0 bytes for IG57]
 
G_M000_IG57:                ;; offset=0x0A05
       mov      eax, r9d
       sar      eax, 31
       and      eax, 7
       add      eax, r9d
       sar      eax, 3
       cdqe     
       shl      rax, 5
       vpand    ymm1, ymm0, ymmword ptr [rax+rcx]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG104
       add      r9d, 8
       lea      eax, [r9+0x08]
       cmp      eax, esi
       jle      SHORT G_M000_IG57
 
G_M000_IG58:                ;; offset=0x0A3A
       mov      dword ptr [rbp-0x160], esi
       cmp      r9d, esi
       jl       G_M000_IG105
       xor      ecx, ecx
       mov      bword ptr [rbp-0x138], rcx
       mov      ecx, 1
 
G_M000_IG59:                ;; offset=0x0A57
       xor      r9, r9
       mov      bword ptr [rbp-0x138], r9
       test     ecx, ecx
       je       G_M000_IG119
       mov      eax, dword ptr [rbp-0x164]
       mov      ecx, eax
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r9, r9
       test     ecx, ecx
       cmovne   r9, bword ptr [rbp-0x188]
       mov      bword ptr [rbp-0x140], r9
       xor      ecx, ecx
       mov      dword ptr [rbp-0x164], eax
       cmp      eax, 8
       jl       SHORT G_M000_IG61
       align    [5 bytes for IG60]
 
G_M000_IG60:                ;; offset=0x0AA0
       mov      eax, ecx
       sar      eax, 31
       and      eax, 7
       add      eax, ecx
       sar      eax, 3
       cdqe     
       shl      rax, 5
       vpand    ymm1, ymm0, ymmword ptr [rax+r9]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG107
       add      ecx, 8
       lea      eax, [rcx+0x08]
       cmp      eax, dword ptr [rbp-0x164]
       jle      SHORT G_M000_IG60
 
G_M000_IG61:                ;; offset=0x0AD6
       cmp      ecx, dword ptr [rbp-0x164]
       jl       G_M000_IG108
       xor      ecx, ecx
       mov      bword ptr [rbp-0x140], rcx
       mov      ecx, 1
 
G_M000_IG62:                ;; offset=0x0AF0
       xor      r9, r9
       mov      bword ptr [rbp-0x140], r9
       test     ecx, ecx
       je       G_M000_IG119
       mov      ecx, r10d
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r9, r9
       test     ecx, ecx
       cmovne   r9, bword ptr [rbp+0x10]
       mov      bword ptr [rbp-0x148], r9
       xor      ecx, ecx
       cmp      r10d, 8
       jl       SHORT G_M000_IG64
       align    [0 bytes for IG63]
 
G_M000_IG63:                ;; offset=0x0B27
       mov      eax, ecx
       sar      eax, 31
       and      eax, 7
       add      eax, ecx
       sar      eax, 3
       cdqe     
       shl      rax, 5
       vpand    ymm1, ymm0, ymmword ptr [rax+r9]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG109
       add      ecx, 8
       lea      eax, [rcx+0x08]
       cmp      eax, r10d
       jle      SHORT G_M000_IG63
 
G_M000_IG64:                ;; offset=0x0B5A
       mov      dword ptr [rbp+0x18], r10d
       cmp      ecx, r10d
       jl       G_M000_IG110
       xor      ecx, ecx
       mov      bword ptr [rbp-0x148], rcx
       mov      ecx, 1
 
G_M000_IG65:                ;; offset=0x0B75
       xor      r9, r9
       mov      bword ptr [rbp-0x148], r9
       test     ecx, ecx
       je       G_M000_IG119
       mov      eax, dword ptr [rbp+0x78]
       lea      r9d, [rax+0x01]
       mov      ecx, r9d
       shr      ecx, 31
       add      r9d, ecx
       sar      r9d, 1
       mov      dword ptr [rbp-0x44], r9d
       mov      ecx, dword ptr [rbp-0x170]
       shr      ecx, 31
       add      ecx, dword ptr [rbp-0x170]
       sar      ecx, 1
       imul     ecx, r9d
       jo       G_M000_IG121
       mov      dword ptr [rbp-0x48], ecx
       xor      eax, eax
       cmp      eax, ecx
       mov      bword ptr [rbp+0x50], r11
       jl       G_M000_IG76
       jmp      G_M000_IG112
 
G_M000_IG66:                ;; offset=0x0BCF
       mov      edi, r11d
       xor      rsi, rsi
       mov      bword ptr [rbp-0x158], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG67:                ;; offset=0x0BE4
       test     edi, edi
       je       SHORT G_M000_IG68
       mov      rsi, r13
 
G_M000_IG68:                ;; offset=0x0BEB
       mov      bword ptr [rbp-0x158], rsi
       xor      edi, edi
       cmp      r11d, 8
       jl       SHORT G_M000_IG70
       align    [6 bytes for IG69]
 
G_M000_IG69:                ;; offset=0x0C00
       mov      r8d, edi
       sar      r8d, 31
       and      r8d, 7
       add      r8d, edi
       sar      r8d, 3
       movsxd   r8, r8d
       shl      r8, 5
       vpand    ymm1, ymm0, ymmword ptr [r8+rsi]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG116
       add      edi, 8
       lea      r8d, [rdi+0x08]
       cmp      r8d, r11d
       jle      SHORT G_M000_IG69
       align    [0 bytes for IG70]
 
G_M000_IG70:                ;; offset=0x0C3A
       mov      dword ptr [rbp+0x48], r11d
       cmp      edi, r11d
       jl       G_M000_IG117
       xor      edi, edi
       mov      bword ptr [rbp-0x158], rdi
       mov      edi, 1
 
G_M000_IG71:                ;; offset=0x0C55
       xor      rsi, rsi
       mov      bword ptr [rbp-0x158], rsi
       test     edi, edi
       je       G_M000_IG119
       mov      r10d, dword ptr [rbp-0x4C]
       mov      dword ptr [rsp], r10d
       mov      edi, dword ptr [rbp-0x50]
       mov      dword ptr [rsp+0x08], edi
       mov      rdi, rdx
       mov      rsi, rcx
       mov      edx, ebx
       mov      ecx, dword ptr [rbp+0x70]
       mov      r8d, dword ptr [rbp+0x78]
       mov      r9d, dword ptr [rbp-0x44]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
       xor      edi, edi
       mov      bword ptr [rbp-0x60], rdi
 
G_M000_IG72:                ;; offset=0x0C94
       mov      bword ptr [rbp-0x68], rdi
 
G_M000_IG73:                ;; offset=0x0C98
       mov      bword ptr [rbp-0x70], rdi
 
G_M000_IG74:                ;; offset=0x0C9C
       mov      bword ptr [rbp-0x78], rdi
       mov      r10d, dword ptr [rbp-0x4C]
       add      r10d, 8
       mov      edi, dword ptr [rbp-0x48]
       cmp      r10d, edi
       mov      eax, r10d
       mov      ecx, edi
       mov      edx, dword ptr [rbp-0x15C]
       mov      r9d, dword ptr [rbp-0x44]
       jge      G_M000_IG93
 
G_M000_IG75:                ;; offset=0x0CC3
       mov      edi, dword ptr [rbp+0x48]
       mov      r8d, dword ptr [rbp+0x58]
 
G_M000_IG76:                ;; offset=0x0CCA
       mov      r11d, ecx
       sub      r11d, eax
       cmp      r11d, 8
       jl       G_M000_IG113
       mov      r11d, 8
       mov      dword ptr [rbp+0x48], edi
       mov      dword ptr [rbp+0x58], r8d
 
G_M000_IG77:                ;; offset=0x0CE7
       mov      dword ptr [rbp-0x50], r11d
       mov      r8d, dword ptr [rbp+0x78]
       mov      dword ptr [rsp], r8d
       mov      dword ptr [rsp+0x08], r9d
       mov      dword ptr [rbp-0x4C], eax
       mov      dword ptr [rsp+0x10], eax
       mov      dword ptr [rsp+0x18], r11d
       mov      rdi, bword ptr [rbp-0x178]
       mov      dword ptr [rbp-0x15C], edx
       mov      esi, edx
       mov      rdx, r12
       mov      ecx, r14d
       mov      r8d, r15d
       mov      r9d, dword ptr [rbp+0x70]
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      edi, r14d
       xor      rsi, rsi
       mov      bword ptr [rbp-0x150], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG78:                ;; offset=0x0D3B
       test     edi, edi
       je       SHORT G_M000_IG79
       mov      rsi, r12
 
G_M000_IG79:                ;; offset=0x0D42
       mov      bword ptr [rbp-0x150], rsi
       xor      edi, edi
       cmp      r14d, 8
       jl       SHORT G_M000_IG81
       align    [0 bytes for IG80]
 
G_M000_IG80:                ;; offset=0x0D51
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
       jne      G_M000_IG114
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, r14d
       jle      SHORT G_M000_IG80
       align    [0 bytes for IG81]
 
G_M000_IG81:                ;; offset=0x0D84
       cmp      edi, r14d
       jl       G_M000_IG115
       xor      edi, edi
       mov      bword ptr [rbp-0x150], rdi
       mov      edi, 1
 
G_M000_IG82:                ;; offset=0x0D9B
       xor      rsi, rsi
       mov      bword ptr [rbp-0x150], rsi
       test     edi, edi
       je       G_M000_IG119
       xor      rdi, rdi
       test     r14d, r14d
       je       SHORT G_M000_IG83
       mov      rdi, r12
 
G_M000_IG83:                ;; offset=0x0DB6
       mov      bword ptr [rbp-0x60], rdi
       xor      rsi, rsi
       cmp      dword ptr [rbp-0x160], 0
       je       SHORT G_M000_IG84
       mov      r10, bword ptr [rbp-0x180]
       mov      rsi, r10
 
G_M000_IG84:                ;; offset=0x0DCF
       mov      bword ptr [rbp-0x68], rsi
       xor      rdx, rdx
       mov      r11d, dword ptr [rbp+0x48]
       test     r11d, r11d
       je       SHORT G_M000_IG85
       mov      bword ptr [rbp+0x40], r13
       mov      rdx, r13
       mov      r13, bword ptr [rbp+0x40]
 
G_M000_IG85:                ;; offset=0x0DE9
       mov      bword ptr [rbp-0x70], rdx
       xor      rcx, rcx
       cmp      dword ptr [rbp+0x58], 0
       je       SHORT G_M000_IG86
       mov      r8, bword ptr [rbp+0x50]
       mov      rcx, r8
 
G_M000_IG86:                ;; offset=0x0DFC
       mov      bword ptr [rbp-0x78], rcx
       mov      qword ptr [rbp-0x58], rcx
       xor      r10d, r10d
       movsxd   r8, ebx
       shl      r8, 2
       mov      rcx, qword ptr [rbp-0x58]
       jmp      SHORT G_M000_IG88
       align    [0 bytes for IG91]
 
G_M000_IG87:                ;; offset=0x0E14
       inc      r10d
       cmp      r10d, 16
       mov      rcx, qword ptr [rbp-0x58]
       mov      r13, bword ptr [rbp+0x40]
       jge      G_M000_IG66
 
G_M000_IG88:                ;; offset=0x0E29
       xor      r9d, r9d
       cmp      r9d, ebx
       mov      bword ptr [rbp+0x40], r13
       jge      SHORT G_M000_IG87
 
G_M000_IG89:                ;; offset=0x0E35
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      eax, r10d
       imul     eax, r15d
       mov      ecx, eax
       imul     ecx, ebx
       movsxd   rcx, ecx
       shl      rcx, 2
       add      rcx, rsi
       movsxd   r13, r9d
       lea      rcx, [rcx+4*r13]
       shl      eax, 3
       cdqe     
       lea      rax, [rdi+4*rax]
       test     r15d, r15d
       jle      SHORT G_M000_IG92
 
G_M000_IG90:                ;; offset=0x0E80
       mov      r13d, r15d
 
G_M000_IG91:                ;; offset=0x0E83
       vmovups  ymm8, ymmword ptr [rcx]
       vbroadcastss ymm9, dword ptr [rax]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rax+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rax+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rax+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rax+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rax+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rax+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rax+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       add      rcx, r8
       add      rax, 32
       dec      r13d
       jne      SHORT G_M000_IG91
 
G_M000_IG92:                ;; offset=0x0EEA
       mov      ecx, r10d
       imul     ecx, ebx
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
       cmp      r9d, ebx
       jl       G_M000_IG89
       jmp      G_M000_IG87
 
G_M000_IG93:                ;; offset=0x0F42
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG119
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG119
       mov      rdi, bword ptr [rbp-0x188]
       mov      esi, dword ptr [rbp-0x164]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG119
       mov      rdi, bword ptr [rbp+0x10]
       mov      esi, dword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG119
       mov      r15, bword ptr [rbp+0x10]
       mov      bword ptr [rsp], r15
       mov      r14d, dword ptr [rbp+0x18]
       mov      dword ptr [rsp+0x08], r14d
       mov      dword ptr [rsp+0x10], ebx
       mov      ebx, dword ptr [rbp+0x70]
       imul     ebx, dword ptr [rbp+0x78]
       mov      dword ptr [rsp+0x18], ebx
       mov      ebx, dword ptr [rbp+0x80]
       mov      dword ptr [rsp+0x20], ebx
       movzx    rdx, byte  ptr [rbp+0x88]
       mov      dword ptr [rsp+0x28], edx
       mov      rdx, bword ptr [rbp+0x20]
       mov      ecx, dword ptr [rbp+0x28]
       mov      r8, bword ptr [rbp-0x188]
       mov      r9d, dword ptr [rbp-0x164]
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG94:                ;; offset=0x0FF8
       vzeroupper 
       add      rsp, 408
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG95:                ;; offset=0x100D
       cmp      dword ptr [rbp+0x80], 16
       je       G_M000_IG03
 
G_M000_IG96:                ;; offset=0x101A
       mov      r8d, 1
       jmp      G_M000_IG04
 
G_M000_IG97:                ;; offset=0x1025
       mov      rdi, 0x76DAAEC7CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0x101D8
       mov      rsi, 0x76DAAF4A9F30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG98:                ;; offset=0x1061
       mov      rdi, 0x76DAAEC7CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0x102B4
       mov      rsi, 0x76DAAF4A9F30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG99:                ;; offset=0x109D
       call     [System.ThrowHelper:ThrowArgumentOutOfRangeException()]
       int3     
 
G_M000_IG100:                ;; offset=0x10A4
       mov      rdi, 0x76DAAEC7CD28
       call     CORINFO_HELP_NEWSFAST
       mov      r12, rax
       mov      edi, 0x102E2
       mov      rsi, 0x76DAAF4A9F30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, r12
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, r12
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG101:                ;; offset=0x10E0
       mov      rdi, 0x76DAAF34A470
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      rdi, rbx
       call     [System.PlatformNotSupportedException:.ctor():this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG102:                ;; offset=0x1104
       xor      ecx, ecx
       jmp      G_M000_IG56
 
G_M000_IG103:                ;; offset=0x110B
       movsxd   rcx, eax
       mov      ecx, dword ptr [r9+4*rcx]
       mov      dword ptr [rbp-0x18C], ecx
       mov      ecx, 0x7F800000
       mov      dword ptr [rbp-0x190], ecx
       mov      ecx, dword ptr [rbp-0x18C]
       andn     ecx, ecx, dword ptr [rbp-0x190]
       je       SHORT G_M000_IG102
       inc      eax
       jmp      G_M000_IG55
 
G_M000_IG104:                ;; offset=0x113B
       xor      ecx, ecx
       mov      dword ptr [rbp-0x160], esi
       jmp      G_M000_IG59
 
G_M000_IG105:                ;; offset=0x1148
       movsxd   rax, r9d
       mov      eax, dword ptr [rcx+4*rax]
       mov      dword ptr [rbp-0x190], eax
       mov      eax, 0x7F800000
       mov      dword ptr [rbp-0x18C], eax
       mov      eax, dword ptr [rbp-0x190]
       andn     eax, eax, dword ptr [rbp-0x18C]
       je       SHORT G_M000_IG106
       inc      r9d
       mov      esi, dword ptr [rbp-0x160]
       jmp      G_M000_IG58
 
G_M000_IG106:                ;; offset=0x117E
       mov      esi, dword ptr [rbp-0x160]
       jmp      SHORT G_M000_IG104
 
G_M000_IG107:                ;; offset=0x1186
       xor      ecx, ecx
       jmp      G_M000_IG62
 
G_M000_IG108:                ;; offset=0x118D
       movsxd   rax, ecx
       mov      eax, dword ptr [r9+4*rax]
       mov      dword ptr [rbp-0x18C], eax
       mov      eax, 0x7F800000
       mov      dword ptr [rbp-0x190], eax
       mov      eax, dword ptr [rbp-0x18C]
       andn     eax, eax, dword ptr [rbp-0x190]
       je       SHORT G_M000_IG107
       inc      ecx
       jmp      G_M000_IG61
 
G_M000_IG109:                ;; offset=0x11BD
       xor      ecx, ecx
       mov      dword ptr [rbp+0x18], r10d
       jmp      G_M000_IG65
 
G_M000_IG110:                ;; offset=0x11C8
       movsxd   rax, ecx
       mov      eax, dword ptr [r9+4*rax]
       mov      dword ptr [rbp-0x190], eax
       mov      eax, 0x7F800000
       mov      dword ptr [rbp-0x18C], eax
       mov      eax, dword ptr [rbp-0x190]
       andn     eax, eax, dword ptr [rbp-0x18C]
       je       SHORT G_M000_IG111
       inc      ecx
       mov      r10d, dword ptr [rbp+0x18]
       jmp      G_M000_IG64
 
G_M000_IG111:                ;; offset=0x11FC
       mov      r10d, dword ptr [rbp+0x18]
       jmp      SHORT G_M000_IG109
 
G_M000_IG112:                ;; offset=0x1202
       mov      dword ptr [rbp+0x58], r8d
       jmp      G_M000_IG93
 
G_M000_IG113:                ;; offset=0x120B
       mov      dword ptr [rbp+0x48], edi
       mov      dword ptr [rbp+0x58], r8d
       jmp      G_M000_IG77
 
G_M000_IG114:                ;; offset=0x1217
       xor      edi, edi
       jmp      G_M000_IG82
 
G_M000_IG115:                ;; offset=0x121E
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      ecx, 0x7F800000
       andn     edx, edx, ecx
       je       SHORT G_M000_IG114
       inc      edi
       jmp      G_M000_IG81
 
G_M000_IG116:                ;; offset=0x1237
       xor      edi, edi
       mov      dword ptr [rbp+0x48], r11d
       jmp      G_M000_IG71
 
G_M000_IG117:                ;; offset=0x1242
       movsxd   r8, edi
       mov      r8d, dword ptr [rsi+4*r8]
       mov      r10d, 0x7F800000
       andn     r8d, r8d, r10d
       je       SHORT G_M000_IG118
       inc      edi
       mov      r11d, dword ptr [rbp+0x48]
       jmp      G_M000_IG70
 
G_M000_IG118:                ;; offset=0x1261
       mov      r11d, dword ptr [rbp+0x48]
       jmp      SHORT G_M000_IG116
 
G_M000_IG119:                ;; offset=0x1267
       xor      eax, eax
 
G_M000_IG120:                ;; offset=0x1269
       vzeroupper 
       add      rsp, 408
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG121:                ;; offset=0x127E
       call     CORINFO_HELP_OVERFLOW
       int3     
 
RWD00  	dd	7F800000h		;       inf

; Total bytes of code 4740

; Assembly listing for method KernelAccess:PrepareWinograd(System.ReadOnlySpan`1[float],int,int,int):float[] (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; rsp based frame
; fully interruptible

G_M000_IG01:                ;; offset=0x0000
       mov      r9d, edx
       mov      edx, esi
       mov      eax, ecx
       mov      r10d, r8d
 
G_M000_IG02:                ;; offset=0x000A
       mov      rsi, rdi
       mov      rcx, 0x76D2C9800190
       mov      r11, gword ptr [rcx]
       mov      ecx, r9d
       mov      r8d, eax
       mov      r9d, r10d
       mov      rdi, gword ptr [r11+0x08]
 
G_M000_IG03:                ;; offset=0x0027
       tail.jmp [r11+0x18]KernelAccess+PrepareCall:Invoke(System.ReadOnlySpan`1[float],int,int,int):float[]:this
 
; Total bytes of code 43

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel256(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 136
       lea      rbp, [rsp+0xB0]
       mov      qword ptr [rbp-0x30], rsi
       mov      qword ptr [rbp-0x38], rdx
       mov      eax, dword ptr [rbp+0x18]
       mov      r11d, dword ptr [rbp+0x20]
       mov      r10d, dword ptr [rbp+0x28]
 
G_M000_IG02:                ;; offset=0x002C
       add      r9d, 2
       mov      dword ptr [rbp-0x44], r9d
       mov      ebx, dword ptr [rbp+0x10]
       add      ebx, 2
       mov      dword ptr [rbp-0x48], ebx
       mov      dword ptr [rbp+0x20], r11d
       mov      dword ptr [rbp+0x28], r10d
       mov      r15d, r11d
       imul     r15d, r10d
       mov      dword ptr [rbp-0x4C], r15d
       mov      r14d, r15d
       sar      r14d, 31
       and      r14d, 7
       add      r14d, r15d
       sar      r14d, 3
       shl      r14d, 3
       mov      dword ptr [rbp-0x50], r14d
       xor      r13d, r13d
       lea      r12d, [8*rcx]
       lea      r12d, [r12+8*r12]
       movsxd   r12, r12d
       shl      r12, 2
       mov      qword ptr [rbp-0x90], r12
       lea      esi, [rax+rax]
       shl      esi, 3
       movsxd   rsi, esi
       lea      r11d, [rax+2*rax]
       shl      r11d, 3
       movsxd   r11, r11d
       lea      r12d, [4*rax]
       shl      r12d, 3
       movsxd   r12, r12d
       lea      r14d, [rax+4*rax]
       shl      r14d, 3
       movsxd   r14, r14d
       mov      dword ptr [rbp-0x40], r8d
       cmp      r13d, r8d
       jl       SHORT G_M000_IG08
 
G_M000_IG03:                ;; offset=0x00BE
       vzeroupper 
       add      rsp, 136
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG04:                ;; offset=0x00D3
       mov      r8d, dword ptr [rbp-0x58]
       imul     r8d, dword ptr [rbp+0x28]
       jmp      G_M000_IG25
       align    [0 bytes for IG05]
 
G_M000_IG05:                ;; offset=0x00E1
       vmulps   ymm3, ymm2, ymm3
       vaddps   ymm0, ymm3, ymm0
       vmulps   ymm2, ymm2, ymm4
       vaddps   ymm1, ymm2, ymm1
       jmp      G_M000_IG15
 
G_M000_IG06:                ;; offset=0x00F6
       mov      r8d, dword ptr [rbp+0x20]
 
G_M000_IG07:                ;; offset=0x00FA
       mov      r13d, dword ptr [rbp-0x54]
       add      r13d, 16
       mov      r10d, dword ptr [rbp-0x40]
       cmp      r13d, r10d
       mov      dword ptr [rbp-0x40], r10d
       mov      dword ptr [rbp+0x20], r8d
       jge      SHORT G_M000_IG03
 
G_M000_IG08:                ;; offset=0x0113
       xor      r10d, r10d
       mov      dword ptr [rbp-0x58], r10d
       mov      dword ptr [rbp-0x54], r13d
       mov      r10d, r13d
       imul     r10d, ecx
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       mov      r13, qword ptr [rbp-0x30]
       lea      r10, [r13+4*r10]
       mov      qword ptr [rbp-0x88], r10
       mov      r13d, dword ptr [rbp-0x58]
       cmp      r13d, dword ptr [rbp+0x20]
       jl       G_M000_IG29
       jmp      SHORT G_M000_IG06
 
G_M000_IG09:                ;; offset=0x014C
       vmulps   ymm3, ymm2, ymm3
       vaddps   ymm0, ymm3, ymm0
       vmulps   ymm2, ymm2, ymm4
       vaddps   ymm1, ymm2, ymm1
       jmp      G_M000_IG17
 
G_M000_IG10:                ;; offset=0x0161
       mov      r13d, dword ptr [rbp-0x68]
       inc      r13d
       cmp      r13d, ecx
       jge      G_M000_IG22
 
G_M000_IG11:                ;; offset=0x0171
       xor      r8d, r8d
       mov      dword ptr [rbp-0x68], r13d
       sar      r13d, 31
       and      r13d, 7
       add      r13d, dword ptr [rbp-0x68]
       sar      r13d, 3
       mov      r9d, dword ptr [rbp-0x44]
       imul     r13d, r9d
       mov      r10d, dword ptr [rbp-0x94]
       add      r13d, r10d
       mov      dword ptr [rbp-0x7C], r13d
       jmp      G_M000_IG19
 
G_M000_IG12:                ;; offset=0x01A3
       vfmadd231ps ymm0, ymm3, ymm2
       vfmadd231ps ymm1, ymm4, ymm2
 
G_M000_IG13:                ;; offset=0x01AD
       add      r15, 32
       add      rdx, 32
       mov      r8d, dword ptr [rbp-0x70]
       inc      r8d
       lea      r8d, [r13+8*r8]
       movsxd   r8, r8d
       vbroadcastss ymm2, dword ptr [rdi+4*r8]
       vmovups  ymm3, ymmword ptr [r15]
       vmovups  ymm4, ymmword ptr [rdx]
       cmp      dword ptr [rbp-0x64], 0
       je       G_M000_IG05
 
G_M000_IG14:                ;; offset=0x01DD
       vfmadd231ps ymm0, ymm3, ymm2
       vfmadd231ps ymm1, ymm4, ymm2
 
G_M000_IG15:                ;; offset=0x01E7
       add      r15, 32
       add      rdx, 32
       mov      r8d, dword ptr [rbp-0x70]
       add      r8d, 2
       lea      r8d, [r13+8*r8]
       movsxd   r8, r8d
       vbroadcastss ymm2, dword ptr [rdi+4*r8]
       vmovups  ymm3, ymmword ptr [r15]
       vmovups  ymm4, ymmword ptr [rdx]
       mov      r8d, dword ptr [rbp-0x64]
       test     r8d, r8d
       je       G_M000_IG09
 
G_M000_IG16:                ;; offset=0x021B
       vfmadd231ps ymm0, ymm3, ymm2
       vfmadd231ps ymm1, ymm4, ymm2
 
G_M000_IG17:                ;; offset=0x0225
       add      r15, 32
       add      rdx, 32
       mov      r13d, dword ptr [rbp-0x6C]
       inc      r13d
       cmp      r13d, 3
       jge      G_M000_IG10
 
G_M000_IG18:                ;; offset=0x023E
       mov      r8d, r13d
       mov      r13d, dword ptr [rbp-0x7C]
 
G_M000_IG19:                ;; offset=0x0245
       mov      dword ptr [rbp-0x6C], r8d
       add      r13d, r8d
       imul     r13d, ebx
       add      r13d, dword ptr [rbp-0x9C]
       mov      dword ptr [rbp-0x70], r13d
       mov      r13d, dword ptr [rbp-0x68]
       sar      r13d, 31
       and      r13d, 7
       add      r13d, dword ptr [rbp-0x68]
       and      r13d, -8
       mov      dword ptr [rbp-0xA4], r13d
       mov      r13d, dword ptr [rbp-0x68]
       sub      r13d, dword ptr [rbp-0xA4]
       mov      r8d, dword ptr [rbp-0x70]
       lea      r8d, [r13+8*r8]
       movsxd   r8, r8d
       vbroadcastss ymm2, dword ptr [rdi+4*r8]
       vmovups  ymm3, ymmword ptr [r15]
       vmovups  ymm4, ymmword ptr [rdx]
       cmp      dword ptr [rbp-0x64], 0
       jne      G_M000_IG12
 
G_M000_IG20:                ;; offset=0x02A6
       vmulps   ymm3, ymm2, ymm3
       vaddps   ymm0, ymm3, ymm0
       vmulps   ymm2, ymm2, ymm4
       vaddps   ymm1, ymm2, ymm1
       jmp      G_M000_IG13
 
G_M000_IG21:                ;; offset=0x02BB
       mov      r9d, dword ptr [rbp-0x44]
       mov      r10d, dword ptr [rbp-0x94]
 
G_M000_IG22:                ;; offset=0x02C6
       mov      r13d, dword ptr [rbp-0x54]
       mov      edx, r13d
       sar      edx, 31
       and      edx, 7
       mov      dword ptr [rbp-0x54], r13d
       add      edx, r13d
       sar      edx, 3
       mov      dword ptr [rbp-0x80], edx
       mov      r15d, dword ptr [rbp-0x4C]
       mov      r8d, edx
       imul     r8d, r15d
       mov      r13d, dword ptr [rbp-0xA0]
       add      r8d, r13d
       mov      r15d, dword ptr [rbp-0x5C]
       add      r8d, r15d
       shl      r8d, 3
       movsxd   r8, r8d
       mov      rdx, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [rdx+4*r8], ymm0
       mov      r8d, dword ptr [rbp-0x54]
       add      r8d, 8
       mov      edx, dword ptr [rbp-0x40]
       cmp      r8d, edx
       jge      SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x031D
       mov      r8d, dword ptr [rbp-0x80]
       inc      r8d
       imul     r8d, dword ptr [rbp-0x4C]
       add      r8d, r13d
       add      r8d, r15d
       shl      r8d, 3
       movsxd   r8, r8d
       mov      rdx, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [rdx+4*r8], ymm1
       mov      qword ptr [rbp-0x38], rdx
 
G_M000_IG24:                ;; offset=0x0344
       inc      r15d
       mov      r8d, r13d
       mov      r13d, r15d
 
G_M000_IG25:                ;; offset=0x034D
       cmp      r13d, dword ptr [rbp+0x28]
       jge      SHORT G_M000_IG27
 
G_M000_IG26:                ;; offset=0x0353
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       mov      r15, qword ptr [rbp-0x88]
       mov      rdx, qword ptr [rbp-0x90]
       add      rdx, r15
       mov      dword ptr [rbp-0xA0], r8d
       mov      dword ptr [rbp-0x5C], r13d
       add      r8d, r13d
       cmp      r8d, dword ptr [rbp-0x50]
       setl     r8b
       movzx    r8, r8b
       mov      dword ptr [rbp-0x64], r8d
       xor      r13d, r13d
       mov      r8d, eax
       imul     r8d, dword ptr [rbp-0x5C]
       mov      dword ptr [rbp-0x9C], r8d
       cmp      r13d, ecx
       jl       G_M000_IG11
       jmp      G_M000_IG21
 
G_M000_IG27:                ;; offset=0x03AA
       mov      r10d, dword ptr [rbp-0x58]
       inc      r10d
       mov      r8d, dword ptr [rbp+0x20]
       cmp      r10d, r8d
       mov      dword ptr [rbp-0x58], r10d
       jge      G_M000_IG07
 
G_M000_IG28:                ;; offset=0x03C2
       mov      dword ptr [rbp+0x20], r8d
 
G_M000_IG29:                ;; offset=0x03C6
       xor      r13d, r13d
       mov      r10d, eax
       imul     r10d, dword ptr [rbp-0x58]
       mov      dword ptr [rbp-0x94], r10d
       jmp      G_M000_IG44
       align    [0 bytes for IG49]
 
G_M000_IG30:                ;; offset=0x03DD
       mov      ecx, dword ptr [rbp-0x3C]
 
G_M000_IG31:                ;; offset=0x03E0
       mov      r13d, dword ptr [rbp-0x54]
       mov      edx, r13d
       sar      edx, 31
       and      edx, 7
       mov      dword ptr [rbp-0x54], r13d
       add      edx, r13d
       sar      edx, 3
       mov      r15d, dword ptr [rbp-0x4C]
       mov      r8d, edx
       imul     r8d, r15d
       mov      r9d, dword ptr [rbp-0xA0]
       add      r8d, r9d
       mov      r10d, dword ptr [rbp-0x5C]
       add      r8d, r10d
       shl      r8d, 3
       mov      dword ptr [rbp-0x74], r8d
       movsxd   r13, r8d
       mov      r15, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [r15+4*r13], ymm0
       mov      r13d, dword ptr [rbp-0x54]
       add      r13d, 8
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG33
 
G_M000_IG32:                ;; offset=0x0439
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       shl      r15d, 3
       movsxd   r15, r15d
       mov      r8, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [r8+4*r15], ymm1
       mov      qword ptr [rbp-0x38], r8
 
G_M000_IG33:                ;; offset=0x045D
       mov      r8d, dword ptr [rbp-0x74]
       lea      r15d, [r8+0x08]
       movsxd   r15, r15d
       mov      r8, qword ptr [rbp-0x38]
       vmovups  ymmword ptr [r8+4*r15], ymm2
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG35
 
G_M000_IG34:                ;; offset=0x047B
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       lea      r15d, [8*r15+0x08]
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm3
 
G_M000_IG35:                ;; offset=0x049B
       mov      r15d, dword ptr [rbp-0x74]
       add      r15d, 16
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm4
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x04B5
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       lea      r15d, [8*r15+0x10]
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm5
 
G_M000_IG37:                ;; offset=0x04D5
       mov      r15d, dword ptr [rbp-0x74]
       add      r15d, 24
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm6
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG39
 
G_M000_IG38:                ;; offset=0x04EF
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       lea      r15d, [8*r15+0x18]
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm7
 
G_M000_IG39:                ;; offset=0x050F
       mov      r15d, dword ptr [rbp-0x74]
       add      r15d, 32
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm8
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG41
 
G_M000_IG40:                ;; offset=0x0529
       lea      r15d, [rdx+0x01]
       imul     r15d, dword ptr [rbp-0x4C]
       add      r15d, r9d
       add      r15d, r10d
       lea      r15d, [8*r15+0x20]
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm9
 
G_M000_IG41:                ;; offset=0x0549
       mov      r15d, dword ptr [rbp-0x74]
       add      r15d, 40
       movsxd   r15, r15d
       vmovups  ymmword ptr [r8+4*r15], ymm10
       mov      r15d, dword ptr [rbp-0x40]
       cmp      r13d, r15d
       jge      SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x0563
       inc      edx
       imul     edx, dword ptr [rbp-0x4C]
       add      r9d, edx
       add      r9d, r10d
       lea      edx, [8*r9+0x28]
       movsxd   rdx, edx
       mov      qword ptr [rbp-0x38], r8
       vmovups  ymmword ptr [r8+4*rdx], ymm11
       mov      r8, qword ptr [rbp-0x38]
 
G_M000_IG43:                ;; offset=0x0588
       add      r10d, 6
       mov      qword ptr [rbp-0x38], r8
       mov      dword ptr [rbp-0x40], r15d
       mov      r13d, r10d
 
G_M000_IG44:                ;; offset=0x0597
       lea      r8d, [r13+0x06]
       cmp      r8d, dword ptr [rbp+0x28]
       jg       G_M000_IG04
 
G_M000_IG45:                ;; offset=0x05A5
       mov      r8d, dword ptr [rbp-0x58]
       imul     r8d, dword ptr [rbp+0x28]
       mov      dword ptr [rbp-0xA0], r8d
       lea      r15d, [r8+r13+0x06]
       cmp      r15d, dword ptr [rbp-0x50]
       jg       G_M000_IG04
 
G_M000_IG46:                ;; offset=0x05C4
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
       mov      r15, qword ptr [rbp-0x88]
       mov      r8, qword ptr [rbp-0x90]
       add      r8, r15
       xor      edx, edx
       mov      dword ptr [rbp-0x5C], r13d
       imul     r13d, eax
       mov      dword ptr [rbp-0x98], r13d
       mov      dword ptr [rbp-0x3C], ecx
       cmp      edx, ecx
       jge      G_M000_IG30
 
G_M000_IG47:                ;; offset=0x0625
       xor      r9d, r9d
       mov      r10d, edx
       sar      r10d, 31
       and      r10d, 7
       add      r10d, edx
       sar      r10d, 3
       imul     r10d, dword ptr [rbp-0x44]
       add      r10d, dword ptr [rbp-0x94]
       mov      dword ptr [rbp-0x78], r10d
       mov      ecx, edx
       sar      ecx, 31
       and      ecx, 7
       add      ecx, edx
       and      ecx, -8
       mov      dword ptr [rbp-0x60], edx
       mov      r13d, edx
       sub      r13d, ecx
       movsxd   rcx, r13d
       shl      rcx, 2
 
G_M000_IG48:                ;; offset=0x0667
       lea      r13d, [r10+r9]
       imul     r13d, ebx
       add      r13d, dword ptr [rbp-0x98]
       shl      r13d, 3
       mov      ebx, 3
 
G_M000_IG49:                ;; offset=0x067F
       vmovups  ymm12, ymmword ptr [r15]
       vmovups  ymm13, ymmword ptr [r8]
       movsxd   r10, r13d
       shl      r10, 2
       add      r10, rdi
       add      r10, rcx
       vbroadcastss ymm14, dword ptr [r10]
       vfmadd231ps ymm0, ymm12, ymm14
       vfmadd231ps ymm1, ymm13, ymm14
       lea      edx, [8*rax]
       movsxd   rdx, edx
       vbroadcastss ymm14, dword ptr [r10+4*rdx]
       vfmadd231ps ymm2, ymm12, ymm14
       vfmadd231ps ymm3, ymm13, ymm14
       vbroadcastss ymm14, dword ptr [r10+4*rsi]
       vfmadd231ps ymm4, ymm12, ymm14
       vfmadd231ps ymm5, ymm13, ymm14
       vbroadcastss ymm14, dword ptr [r10+4*r11]
       vfmadd231ps ymm6, ymm12, ymm14
       vfmadd231ps ymm7, ymm13, ymm14
       vbroadcastss ymm14, dword ptr [r10+4*r12]
       vfmadd231ps ymm8, ymm12, ymm14
       vfmadd231ps ymm9, ymm13, ymm14
       vbroadcastss ymm14, dword ptr [r10+4*r14]
       vfmadd231ps ymm10, ymm12, ymm14
       vfmadd231ps ymm11, ymm13, ymm14
       add      r15, 32
       add      r8, 32
       add      r13d, 8
       dec      ebx
       jne      G_M000_IG49
 
G_M000_IG50:                ;; offset=0x0713
       inc      r9d
       cmp      r9d, 3
       mov      ebx, dword ptr [rbp-0x48]
       mov      r10d, dword ptr [rbp-0x78]
       jl       G_M000_IG48
 
G_M000_IG51:                ;; offset=0x0727
       mov      edx, dword ptr [rbp-0x60]
       inc      edx
       mov      ecx, dword ptr [rbp-0x3C]
       cmp      edx, ecx
       jge      G_M000_IG31
 
G_M000_IG52:                ;; offset=0x0737
       mov      dword ptr [rbp-0x3C], ecx
       jmp      G_M000_IG47
 
; Total bytes of code 1855

