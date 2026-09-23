; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x3af
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 101
; 45 inlinees with PGO data; 165 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 224
       mov      qword ptr [rsp+0x578], r15
       mov      qword ptr [rsp+0x570], r14
       mov      qword ptr [rsp+0x568], r13
       mov      qword ptr [rsp+0x560], r12
       mov      qword ptr [rsp+0x558], rbx
       lea      rbp, [rsp+0xE0]
       xor      eax, eax
       mov      qword ptr [rbp-0x30], rax
       mov      qword ptr [rbp-0x38], rax
       mov      r15d, dword ptr [rbp+0x500]
       mov      ebx, dword ptr [rbp+0x508]
       mov      r13d, dword ptr [rbp+0x510]
       mov      r14d, dword ptr [rbp+0x518]
       mov      r12d, dword ptr [rbp+0x520]
       mov      eax, dword ptr [rbp+0x42C]
       mov      r11d, dword ptr [rbp+0x428]
       mov      r10d, dword ptr [rbp+0x424]
 
G_M000_IG02:                ;; offset=0x007C
       mov      r9, bword ptr [rbp+0x458]
       mov      bword ptr [rbp-0x70], r9
       mov      r8d, dword ptr [rbp+0x460]
       mov      dword ptr [rbp-0x44], r8d
       mov      rcx, bword ptr [rbp+0x4D0]
       mov      bword ptr [rbp-0x78], rcx
       mov      edx, dword ptr [rbp+0x4D8]
       mov      dword ptr [rbp-0x48], edx
       mov      rsi, bword ptr [rbp+0x4E0]
       mov      bword ptr [rbp-0x80], rsi
       mov      edi, dword ptr [rbp+0x4E8]
       mov      dword ptr [rbp-0x4C], edi
       mov      r9, bword ptr [rbp+0x4F0]
       mov      bword ptr [rbp-0x88], r9
       mov      r8d, dword ptr [rbp+0x4F8]
       mov      dword ptr [rbp-0x50], r8d
       mov      r9, bword ptr [rbp+0x468]
       mov      bword ptr [rbp-0x90], r9
       mov      r9d, dword ptr [rbp+0x470]
       mov      dword ptr [rbp-0x54], r9d
       mov      r9, bword ptr [rbp+0x448]
       mov      bword ptr [rbp-0x98], r9
       mov      r9d, dword ptr [rbp+0x450]
       mov      dword ptr [rbp-0x58], r9d
       mov      r9, bword ptr [rbp+0x4B0]
       mov      bword ptr [rbp-0xA0], r9
       mov      r9d, dword ptr [rbp+0x4B8]
       mov      dword ptr [rbp-0x5C], r9d
       mov      r9, bword ptr [rbp+0x4C0]
       mov      bword ptr [rbp-0xA8], r9
       mov      r9d, dword ptr [rbp+0x4C8]
       mov      dword ptr [rbp-0x60], r9d
       mov      r9d, r13d
       imul     r9d, r14d
       mov      dword ptr [rbp-0x64], r9d
       cmp      r10d, r11d
       jl       G_M000_IG23
       jmp      G_M000_IG50
 
G_M000_IG03:                ;; offset=0x0150
       inc      edi
       cmp      edi, 16
       jge      G_M000_IG10
 
G_M000_IG04:                ;; offset=0x015B
       xor      r9d, r9d
       cmp      r9d, ebx
       jge      SHORT G_M000_IG03
 
G_M000_IG05:                ;; offset=0x0163
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r12d, edi
       imul     r12d, r15d
       mov      r11d, r12d
       imul     r11d, ebx
       movsxd   r11, r11d
       shl      r11, 2
       add      r11, rcx
       movsxd   r13, r9d
       lea      r11, [r11+4*r13]
       shl      r12d, 3
       movsxd   r13, r12d
       lea      r13, [rsi+4*r13]
       test     r15d, r15d
       jle      SHORT G_M000_IG08
 
G_M000_IG06:                ;; offset=0x01B2
       mov      r12d, r15d
       align    [0 bytes for IG07]
 
G_M000_IG07:                ;; offset=0x01B5
       vmovups  ymm8, ymmword ptr [r11]
       vbroadcastss ymm9, dword ptr [r13]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r13+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r13+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r13+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r13+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r13+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r13+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r13+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       add      r11, rdx
       add      r13, 32
       dec      r12d
       jne      SHORT G_M000_IG07
 
G_M000_IG08:                ;; offset=0x021E
       mov      r11d, edi
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
       jl       G_M000_IG05
       jmp      G_M000_IG03
 
G_M000_IG09:                ;; offset=0x0280
       mov      rsi, rdx
       mov      rdx, r8
       mov      ecx, r15d
       mov      r8d, ebx
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd512(ptr,ptr,ptr,int,int)]
       mov      r8, qword ptr [rbp+0x408]
 
G_M000_IG10:                ;; offset=0x0299
       mov      r10d, dword ptr [rbp-0x4C]
       mov      edx, r10d
       xor      rdi, rdi
       mov      bword ptr [rbp-0x38], rdi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG11:                ;; offset=0x02AF
       test     edx, edx
       je       SHORT G_M000_IG12
       mov      r11, bword ptr [rbp-0x80]
       mov      rdi, r11
 
G_M000_IG12:                ;; offset=0x02BA
       mov      bword ptr [rbp-0x38], rdi
       xor      edx, edx
       cmp      r10d, 8
       jl       SHORT G_M000_IG14
       align    [0 bytes for IG13]
 
G_M000_IG13:                ;; offset=0x02C6
       mov      esi, edx
       sar      esi, 31
       and      esi, 7
       add      esi, edx
       sar      esi, 3
       movsxd   rsi, esi
       shl      rsi, 5
       vpand    ymm1, ymm0, ymmword ptr [rsi+rdi]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG54
       add      edx, 8
       lea      esi, [rdx+0x08]
       cmp      esi, r10d
       jle      SHORT G_M000_IG13
       align    [0 bytes for IG14]
 
G_M000_IG14:                ;; offset=0x02F9
       cmp      edx, r10d
       jl       G_M000_IG55
       xor      edx, edx
       mov      bword ptr [rbp-0x38], rdx
       mov      edx, 1
 
G_M000_IG15:                ;; offset=0x030D
       xor      rdi, rdi
       mov      bword ptr [rbp-0x38], rdi
       test     edx, edx
       je       G_M000_IG57
       cmp      dword ptr [rbp+0x520], 16
       je       G_M000_IG47
 
G_M000_IG16:                ;; offset=0x0328
       mov      r9d, dword ptr [rbp-0x64]
       mov      dword ptr [rbp-0x3C], r9d
       xor      esi, esi
       cmp      esi, ebx
       jl       G_M000_IG38
 
G_M000_IG17:                ;; offset=0x033A
       mov      r11d, dword ptr [rbp+0x424]
       mov      r13d, dword ptr [rbp+0x510]
 
G_M000_IG18:                ;; offset=0x0348
       xor      edi, edi
       mov      bword ptr [rbp+0x3F8], rdi
 
G_M000_IG19:                ;; offset=0x0351
       mov      bword ptr [rbp+0x3F0], rdi
 
G_M000_IG20:                ;; offset=0x0358
       mov      bword ptr [rbp+0x3E8], rdi
 
G_M000_IG21:                ;; offset=0x035F
       mov      bword ptr [rbp+0x3E0], rdi
       add      r11d, 8
       mov      edi, dword ptr [rbp+0x428]
       cmp      r11d, edi
       mov      r10d, r11d
       mov      eax, dword ptr [rbp+0x42C]
       mov      rcx, bword ptr [rbp-0x78]
       jge      G_M000_IG48
 
G_M000_IG22:                ;; offset=0x0386
       mov      r11d, edi
       mov      r12d, dword ptr [rbp+0x520]
 
G_M000_IG23:                ;; offset=0x0390
       mov      dword ptr [rbp+0x428], r11d
       mov      r9d, r11d
       sub      r9d, r10d
       cmp      r9d, 8
       jl       G_M000_IG51
       mov      r9d, 8
 
G_M000_IG24:                ;; offset=0x03AD
       mov      dword ptr [rbp+0x420], r9d
       mov      dword ptr [rsp], r14d
       mov      dword ptr [rbp+0x42C], eax
       mov      dword ptr [rsp+0x08], eax
       mov      dword ptr [rbp+0x424], r10d
       mov      dword ptr [rsp+0x10], r10d
       mov      dword ptr [rsp+0x18], r9d
       mov      rdi, bword ptr [rbp-0x90]
       mov      esi, dword ptr [rbp-0x54]
       mov      rdx, rcx
       mov      ecx, dword ptr [rbp-0x48]
       mov      r8d, r15d
       mov      dword ptr [rbp+0x510], r13d
       mov      r9d, r13d
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      eax, dword ptr [rbp-0x48]
       mov      edi, eax
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG25:                ;; offset=0x040A
       test     edi, edi
       je       SHORT G_M000_IG26
       mov      r9, bword ptr [rbp-0x78]
       mov      rsi, r9
 
G_M000_IG26:                ;; offset=0x0415
       mov      bword ptr [rbp-0x30], rsi
       xor      edi, edi
       cmp      eax, 8
       jl       SHORT G_M000_IG28
       align    [0 bytes for IG27]
 
G_M000_IG27:                ;; offset=0x0420
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
       jne      G_M000_IG52
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, eax
       jle      SHORT G_M000_IG27
       align    [0 bytes for IG28]
 
G_M000_IG28:                ;; offset=0x0452
       cmp      edi, eax
       jl       G_M000_IG53
       xor      edi, edi
       mov      bword ptr [rbp-0x30], rdi
       mov      edi, 1
 
G_M000_IG29:                ;; offset=0x0465
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       test     edi, edi
       je       G_M000_IG57
       xor      rdi, rdi
       test     eax, eax
       je       SHORT G_M000_IG30
       mov      r9, bword ptr [rbp-0x78]
       mov      rdi, r9
 
G_M000_IG30:                ;; offset=0x0480
       mov      bword ptr [rbp+0x3F8], rdi
       mov      rsi, rdi
       xor      rdx, rdx
       cmp      dword ptr [rbp-0x44], 0
       je       SHORT G_M000_IG31
       mov      r11, bword ptr [rbp-0x70]
       mov      rdx, r11
 
G_M000_IG31:                ;; offset=0x0499
       mov      bword ptr [rbp+0x3F0], rdx
       mov      rcx, rdx
       xor      r8, r8
       cmp      dword ptr [rbp-0x4C], 0
       je       SHORT G_M000_IG32
       mov      r8, bword ptr [rbp-0x80]
       mov      r11, r8
       mov      r8, r11
 
G_M000_IG32:                ;; offset=0x04B6
       mov      bword ptr [rbp+0x3E8], r8
       mov      qword ptr [rbp+0x408], r8
       xor      r11, r11
       cmp      dword ptr [rbp-0x50], 0
       je       SHORT G_M000_IG33
       mov      r11, bword ptr [rbp-0x88]
 
G_M000_IG33:                ;; offset=0x04D4
       mov      bword ptr [rbp+0x3E0], r11
       mov      qword ptr [rbp+0x400], r11
       mov      dword ptr [rbp+0x520], r12d
       cmp      r12d, 16
       je       G_M000_IG09
 
G_M000_IG34:                ;; offset=0x04F3
       xor      edi, edi
       movsxd   rdx, ebx
       shl      rdx, 2
       jmp      G_M000_IG04
       align    [0 bytes for IG39]
 
G_M000_IG35:                ;; offset=0x0501
       mov      eax, dword ptr [rbp-0x3C]
       mov      edi, dword ptr [rbp+0x420]
       mov      r11d, dword ptr [rbp+0x424]
       mov      r12, qword ptr [rbp+0x400]
       mov      r13d, dword ptr [rbp+0x510]
 
G_M000_IG36:                ;; offset=0x051F
       add      esi, 8
       cmp      esi, ebx
       jge      G_M000_IG18
 
G_M000_IG37:                ;; offset=0x052A
       mov      dword ptr [rbp+0x424], r11d
       mov      dword ptr [rbp+0x510], r13d
 
G_M000_IG38:                ;; offset=0x0538
       xor      ecx, ecx
       cmp      ecx, dword ptr [rbp+0x420]
       jge      SHORT G_M000_IG35
 
G_M000_IG39:                ;; offset=0x0542
       lea      r9d, [8*rsi]
       movsxd   r9, r9d
       shl      r9, 2
       add      r9, r8
       lea      r12d, [8*rcx]
       movsxd   r12, r12d
       lea      r9, [r9+4*r12]
       lea      r12d, [8*rbx]
       mov      r11d, dword ptr [rbp+0x424]
       lea      r13d, [r11+rcx]
       mov      eax, r13d
       cdq      
       idiv     edx:eax, dword ptr [rbp+0x42C]
       lea      edx, [rax+rax]
       mov      dword ptr [rbp-0x40], edx
       mov      eax, r13d
       cdq      
       idiv     edx:eax, dword ptr [rbp+0x42C]
       add      edx, edx
       vmovups  ymm0, ymmword ptr [r9]
       lea      eax, [4*r12]
       cdqe     
       vmovups  ymm1, ymmword ptr [r9+4*rax]
       vaddps   ymm0, ymm0, ymm1
       lea      eax, [8*r12]
       cdqe     
       vmovups  ymm2, ymmword ptr [r9+4*rax]
       vaddps   ymm0, ymm0, ymm2
       vsubps   ymm1, ymm1, ymm2
       lea      eax, [r12+2*r12]
       lea      r13d, [4*rax]
       movsxd   r13, r13d
       vsubps   ymm1, ymm1, ymmword ptr [r9+4*r13]
       movsxd   r13, r12d
       vmovups  ymm2, ymmword ptr [r9+4*r13]
       lea      r13d, [r12+4*r12]
       movsxd   rdi, r13d
       vmovups  ymm3, ymmword ptr [r9+4*rdi]
       vaddps   ymm2, ymm2, ymm3
       lea      edi, [r12+8*r12]
       movsxd   rdi, edi
       vmovups  ymm4, ymmword ptr [r9+4*rdi]
       vaddps   ymm2, ymm2, ymm4
       vsubps   ymm3, ymm3, ymm4
       imul     edi, r12d, 13
       movsxd   rdi, edi
       vsubps   ymm3, ymm3, ymmword ptr [r9+4*rdi]
       lea      edi, [r12+r12]
       movsxd   rdi, edi
       vmovups  ymm4, ymmword ptr [r9+4*rdi]
       lea      edi, [rax+rax]
       movsxd   rdi, edi
       vmovups  ymm5, ymmword ptr [r9+4*rdi]
       vaddps   ymm4, ymm4, ymm5
       add      r13d, r13d
       movsxd   rdi, r13d
       vmovups  ymm6, ymmword ptr [r9+4*rdi]
       vaddps   ymm4, ymm6, ymm4
       vsubps   ymm5, ymm5, ymm6
       imul     edi, r12d, 14
       movsxd   rdi, edi
       vsubps   ymm5, ymm5, ymmword ptr [r9+4*rdi]
       movsxd   rdi, eax
       vmovups  ymm6, ymmword ptr [r9+4*rdi]
       lea      edi, [8*r12]
       sub      edi, r12d
 
G_M000_IG40:                ;; offset=0x0666
       movsxd   rdi, edi
       vmovups  ymm7, ymmword ptr [r9+4*rdi]
       vaddps   ymm6, ymm6, ymm7
       imul     edi, r12d, 11
       movsxd   rdi, edi
       vmovups  ymm8, ymmword ptr [r9+4*rdi]
       vaddps   ymm6, ymm6, ymm8
       vsubps   ymm7, ymm7, ymm8
       mov      edi, r12d
       shl      edi, 4
       sub      edi, r12d
       movsxd   rdi, edi
       vsubps   ymm7, ymm7, ymmword ptr [r9+4*rdi]
       vaddps   ymm0, ymm2, ymm0
       vaddps   ymm0, ymm0, ymm4
       vsubps   ymm2, ymm2, ymm4
       vsubps   ymm2, ymm2, ymm6
       vaddps   ymm1, ymm1, ymm3
       vaddps   ymm1, ymm1, ymm5
       vsubps   ymm3, ymm3, ymm5
       vsubps   ymm3, ymm3, ymm7
       mov      edi, dword ptr [rbp-0x40]
       mov      r9d, edi
       imul     r9d, r14d
       add      r9d, edx
       shl      r9d, 3
       movsxd   r9, r9d
       mov      eax, dword ptr [rbp-0x3C]
       mov      r13d, esi
       imul     r13d, eax
       movsxd   r13, r13d
       shl      r13, 2
       mov      r12, qword ptr [rbp+0x400]
       add      r13, r12
       lea      r9, [r13+4*r9]
       vmovups  ymmword ptr [r9], ymm0
       lea      r13d, [rdx+0x01]
       cmp      r13d, r14d
       jge      SHORT G_M000_IG42
 
G_M000_IG41:                ;; offset=0x06FE
       vmovups  ymmword ptr [r9+0x20], ymm2
 
G_M000_IG42:                ;; offset=0x0704
       inc      edi
       mov      r13d, dword ptr [rbp+0x510]
       cmp      edi, r13d
       jge      SHORT G_M000_IG45
 
G_M000_IG43:                ;; offset=0x0712
       lea      edi, [8*r14]
       movsxd   rdi, edi
       vmovups  ymmword ptr [r9+4*rdi], ymm1
       inc      edx
       cmp      edx, r14d
       jge      SHORT G_M000_IG45
 
G_M000_IG44:                ;; offset=0x072A
       lea      edi, [8*r14+0x08]
       movsxd   rdi, edi
       vmovups  ymmword ptr [r9+4*rdi], ymm3
 
G_M000_IG45:                ;; offset=0x073B
       inc      ecx
       mov      edi, dword ptr [rbp+0x420]
       cmp      ecx, edi
       jge      G_M000_IG36
 
G_M000_IG46:                ;; offset=0x074B
       mov      dword ptr [rbp+0x424], r11d
       mov      dword ptr [rbp+0x510], r13d
       jmp      G_M000_IG39
 
G_M000_IG47:                ;; offset=0x075E
       mov      r13d, dword ptr [rbp+0x510]
       mov      r9d, dword ptr [rbp+0x424]
       mov      dword ptr [rsp], r9d
       mov      edi, dword ptr [rbp+0x420]
       mov      dword ptr [rsp+0x08], edi
       mov      rdi, r8
       mov      rsi, qword ptr [rbp+0x400]
       mov      edx, ebx
       mov      ecx, r13d
       mov      r8d, r14d
       mov      r9d, dword ptr [rbp+0x42C]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd512(ptr,ptr,int,int,int,int,int,int)]
       mov      r11d, dword ptr [rbp+0x424]
       jmp      G_M000_IG18
 
G_M000_IG48:                ;; offset=0x07A5
       mov      rdi, bword ptr [rbp-0x88]
       mov      esi, dword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG57
       mov      rdi, bword ptr [rbp-0x88]
       mov      esi, dword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG57
       mov      rdi, bword ptr [rbp-0x98]
       mov      esi, dword ptr [rbp-0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG57
       mov      rdi, bword ptr [rbp-0xA0]
       mov      esi, dword ptr [rbp-0x5C]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG57
       lea      rdi, [rsp]
       lea      rsi, [rbp+0x4B0]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       mov      dword ptr [rsp+0x10], ebx
       mov      ebx, dword ptr [rbp-0x64]
       mov      dword ptr [rsp+0x18], ebx
       mov      r12d, dword ptr [rbp+0x520]
       mov      dword ptr [rsp+0x20], r12d
       movzx    r8, byte  ptr [rbp+0x528]
       mov      dword ptr [rsp+0x28], r8d
       mov      r8, bword ptr [rbp-0x98]
       mov      r9d, dword ptr [rbp-0x58]
       mov      rdx, bword ptr [rbp-0xA8]
       mov      ecx, dword ptr [rbp-0x60]
       mov      rdi, bword ptr [rbp-0x88]
       mov      esi, dword ptr [rbp-0x50]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG49:                ;; offset=0x086F
       vzeroupper 
       add      rsp, 0x558
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG50:                ;; offset=0x0884
       mov      dword ptr [rbp+0x520], r12d
       jmp      G_M000_IG48
 
G_M000_IG51:                ;; offset=0x0890
       jmp      G_M000_IG24
 
G_M000_IG52:                ;; offset=0x0895
       xor      edi, edi
       jmp      G_M000_IG29
 
G_M000_IG53:                ;; offset=0x089C
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      ecx, 0x7F800000
       andn     edx, edx, ecx
       je       SHORT G_M000_IG52
       inc      edi
       jmp      G_M000_IG28
 
G_M000_IG54:                ;; offset=0x08B5
       xor      edx, edx
       jmp      G_M000_IG15
 
G_M000_IG55:                ;; offset=0x08BC
       movsxd   rsi, edx
       mov      esi, dword ptr [rdi+4*rsi]
       mov      ecx, 0x7F800000
       andn     esi, esi, ecx
       je       SHORT G_M000_IG56
       inc      edx
       mov      r10d, dword ptr [rbp-0x4C]
       jmp      G_M000_IG14
 
G_M000_IG56:                ;; offset=0x08D9
       mov      r10d, dword ptr [rbp-0x4C]
       jmp      SHORT G_M000_IG54
 
G_M000_IG57:                ;; offset=0x08DF
       xor      eax, eax
 
G_M000_IG58:                ;; offset=0x08E1
       vzeroupper 
       add      rsp, 0x558
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
RWD00  	dd	7F800000h		;       inf

; Total bytes of code 2294

