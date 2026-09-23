; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
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
       mov      qword ptr [rsp+0x568], r15
       mov      qword ptr [rsp+0x560], r14
       mov      qword ptr [rsp+0x558], r13
       mov      qword ptr [rsp+0x550], r12
       mov      qword ptr [rsp+0x548], rbx
       lea      rbp, [rsp+0xE0]
       xor      eax, eax
       mov      qword ptr [rbp-0x30], rax
       mov      qword ptr [rbp-0x38], rax
       mov      r15d, dword ptr [rbp+0x4F0]
       mov      ebx, dword ptr [rbp+0x4F8]
       mov      r13d, dword ptr [rbp+0x500]
       mov      r14d, dword ptr [rbp+0x508]
       mov      r12d, dword ptr [rbp+0x510]
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
       mov      r8d, r13d
       imul     r8d, r14d
       mov      dword ptr [rbp-0x64], r8d
       cmp      r11d, r9d
       jl       G_M000_IG15
       jmp      G_M000_IG46
 
G_M000_IG03:                ;; offset=0x014B
       mov      r10d, dword ptr [rbp-0x50]
       mov      edx, r10d
       xor      rdi, rdi
       mov      bword ptr [rbp-0x38], rdi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG04:                ;; offset=0x0161
       test     edx, edx
       je       SHORT G_M000_IG05
       mov      r11, bword ptr [rbp-0x88]
       mov      rdi, r11
 
G_M000_IG05:                ;; offset=0x016F
       mov      bword ptr [rbp-0x38], rdi
       xor      edx, edx
       cmp      r10d, 8
       jl       SHORT G_M000_IG07
       align    [5 bytes for IG06]
 
G_M000_IG06:                ;; offset=0x0180
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
       jne      G_M000_IG51
       add      edx, 8
       lea      esi, [rdx+0x08]
       cmp      esi, r10d
       jle      SHORT G_M000_IG06
       align    [0 bytes for IG07]
 
G_M000_IG07:                ;; offset=0x01B3
       cmp      edx, r10d
       jl       G_M000_IG52
       xor      edx, edx
       mov      bword ptr [rbp-0x38], rdx
       mov      edx, 1
 
G_M000_IG08:                ;; offset=0x01C7
       xor      rdi, rdi
       mov      bword ptr [rbp-0x38], rdi
       test     edx, edx
       je       G_M000_IG55
       cmp      dword ptr [rbp+0x510], 16
       jne      G_M000_IG54
       mov      r9d, dword ptr [rbp-0x64]
       mov      dword ptr [rbp-0x3C], r9d
       xor      esi, esi
       cmp      esi, ebx
       jl       G_M000_IG37
 
G_M000_IG09:                ;; offset=0x01F4
       mov      r11d, dword ptr [rbp+0x500]
       mov      r12d, dword ptr [rbp+0x414]
 
G_M000_IG10:                ;; offset=0x0202
       xor      edi, edi
       mov      bword ptr [rbp+0x3E8], rdi
 
G_M000_IG11:                ;; offset=0x020B
       mov      bword ptr [rbp+0x3E0], rdi
 
G_M000_IG12:                ;; offset=0x0212
       mov      bword ptr [rbp+0x3D8], rdi
 
G_M000_IG13:                ;; offset=0x0219
       mov      bword ptr [rbp+0x3D0], rdi
       add      r12d, 8
       mov      r13d, dword ptr [rbp+0x418]
       cmp      r12d, r13d
       mov      r9d, r13d
       mov      eax, dword ptr [rbp-0x4C]
       mov      r10d, dword ptr [rbp+0x41C]
       jge      G_M000_IG32
 
G_M000_IG14:                ;; offset=0x0241
       mov      r13d, r11d
       mov      r11d, r12d
       mov      r12d, dword ptr [rbp+0x510]
 
G_M000_IG15:                ;; offset=0x024E
       mov      dword ptr [rbp+0x418], r9d
       mov      r8d, r9d
       sub      r8d, r11d
       cmp      r8d, 8
       jl       G_M000_IG47
       mov      r8d, 8
 
G_M000_IG16:                ;; offset=0x026B
       mov      dword ptr [rbp+0x410], r8d
       mov      dword ptr [rsp], r14d
       mov      dword ptr [rbp+0x41C], r10d
       mov      dword ptr [rsp+0x08], r10d
       mov      dword ptr [rbp+0x414], r11d
       mov      dword ptr [rsp+0x10], r11d
       mov      dword ptr [rsp+0x18], r8d
       mov      rdi, bword ptr [rbp-0x98]
       mov      esi, dword ptr [rbp-0x58]
       mov      rdx, bword ptr [rbp-0x80]
       mov      ecx, eax
       mov      r8d, r15d
       mov      dword ptr [rbp+0x500], r13d
       mov      r9d, r13d
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      eax, dword ptr [rbp-0x4C]
       mov      edi, eax
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG17:                ;; offset=0x02CA
       test     edi, edi
       je       SHORT G_M000_IG18
       mov      r9, bword ptr [rbp-0x80]
       mov      rsi, r9
 
G_M000_IG18:                ;; offset=0x02D5
       mov      bword ptr [rbp-0x30], rsi
       xor      edi, edi
       cmp      eax, 8
       jl       SHORT G_M000_IG20
       align    [0 bytes for IG19]
 
G_M000_IG19:                ;; offset=0x02E0
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
       jne      G_M000_IG48
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, eax
       jle      SHORT G_M000_IG19
       align    [0 bytes for IG20]
 
G_M000_IG20:                ;; offset=0x0312
       cmp      edi, eax
       jl       G_M000_IG49
       xor      edi, edi
       mov      bword ptr [rbp-0x30], rdi
       mov      edi, 1
 
G_M000_IG21:                ;; offset=0x0325
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       test     edi, edi
       je       G_M000_IG55
       xor      rdi, rdi
       test     eax, eax
       je       SHORT G_M000_IG22
       mov      r9, bword ptr [rbp-0x80]
       mov      rdi, r9
 
G_M000_IG22:                ;; offset=0x0340
       mov      bword ptr [rbp+0x3E8], rdi
       mov      rsi, rdi
       xor      rdx, rdx
       cmp      dword ptr [rbp-0x44], 0
       je       SHORT G_M000_IG23
       mov      r11, bword ptr [rbp-0x70]
       mov      rdx, r11
 
G_M000_IG23:                ;; offset=0x0359
       mov      bword ptr [rbp+0x3E0], rdx
       mov      rcx, rdx
       xor      r8, r8
       cmp      dword ptr [rbp-0x50], 0
       je       SHORT G_M000_IG24
       mov      r8, bword ptr [rbp-0x88]
       mov      r11, r8
       mov      r8, r11
 
G_M000_IG24:                ;; offset=0x0379
       mov      bword ptr [rbp+0x3D8], r8
       mov      qword ptr [rbp+0x3F8], r8
       xor      r11, r11
       cmp      dword ptr [rbp-0x54], 0
       je       SHORT G_M000_IG25
       mov      r11, bword ptr [rbp-0x90]
 
G_M000_IG25:                ;; offset=0x0397
       mov      bword ptr [rbp+0x3D0], r11
       mov      qword ptr [rbp+0x3F0], r11
       mov      dword ptr [rbp+0x510], r12d
       cmp      r12d, 16
       jne      G_M000_IG50
       xor      edi, edi
       movsxd   rdx, ebx
       shl      rdx, 2
       jmp      G_M000_IG31
       align    [0 bytes for IG28]
 
G_M000_IG26:                ;; offset=0x03C4
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
       jle      SHORT G_M000_IG29
 
G_M000_IG27:                ;; offset=0x0413
       mov      r12d, r15d
 
G_M000_IG28:                ;; offset=0x0416
       vmovups  zmm8, zmmword ptr [r11]
       vfmadd231ps zmm0, zmm8, dword ptr [r13] {1to16}
       vfmadd231ps zmm1, zmm8, dword ptr [r13+0x04] {1to16}
       vfmadd231ps zmm2, zmm8, dword ptr [r13+0x08] {1to16}
       vfmadd231ps zmm3, zmm8, dword ptr [r13+0x0C] {1to16}
       vfmadd231ps zmm4, zmm8, dword ptr [r13+0x10] {1to16}
       vfmadd231ps zmm5, zmm8, dword ptr [r13+0x14] {1to16}
       vfmadd231ps zmm6, zmm8, dword ptr [r13+0x18] {1to16}
       vfmadd231ps zmm7, zmm8, dword ptr [r13+0x1C] {1to16}
       add      r11, rdx
       add      r13, 32
       dec      r12d
       jne      SHORT G_M000_IG28
 
G_M000_IG29:                ;; offset=0x0460
       mov      r11d, edi
       imul     r11d, ebx
       add      r11d, r9d
       shl      r11d, 3
       movsxd   r11, r11d
       lea      r11, [r8+4*r11]
       vmovups  zmmword ptr [r11], zmm0
       vmovups  zmmword ptr [r11+0x40], zmm1
       vmovups  zmmword ptr [r11+0x80], zmm2
       vmovups  zmmword ptr [r11+0xC0], zmm3
       vmovups  zmmword ptr [r11+0x100], zmm4
       vmovups  zmmword ptr [r11+0x140], zmm5
       vmovups  zmmword ptr [r11+0x180], zmm6
       vmovups  zmmword ptr [r11+0x1C0], zmm7
       add      r9d, 16
       cmp      r9d, ebx
       jl       G_M000_IG26
 
G_M000_IG30:                ;; offset=0x04B9
       inc      edi
       cmp      edi, 16
       jge      G_M000_IG03
 
G_M000_IG31:                ;; offset=0x04C4
       xor      r9d, r9d
       cmp      r9d, ebx
       jl       G_M000_IG26
       jmp      SHORT G_M000_IG30
 
G_M000_IG32:                ;; offset=0x04D2
       mov      rdi, bword ptr [rbp-0x90]
       mov      esi, dword ptr [rbp-0x54]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG55
       mov      rdi, bword ptr [rbp-0x90]
       mov      esi, dword ptr [rbp-0x54]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG55
       mov      rdi, bword ptr [rbp-0x78]
       mov      esi, dword ptr [rbp-0x48]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG55
       mov      rdi, bword ptr [rbp-0xA0]
       mov      esi, dword ptr [rbp-0x5C]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG55
       lea      rdi, [rsp]
       lea      rsi, [rbp+0x4A0]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       mov      dword ptr [rsp+0x10], ebx
       mov      ebx, dword ptr [rbp-0x64]
       mov      dword ptr [rsp+0x18], ebx
       mov      r12d, dword ptr [rbp+0x510]
       mov      dword ptr [rsp+0x20], r12d
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
 
G_M000_IG33:                ;; offset=0x0596
       vzeroupper 
       add      rsp, 0x548
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG34:                ;; offset=0x05AB
       mov      eax, dword ptr [rbp-0x3C]
       mov      edi, dword ptr [rbp+0x410]
       mov      r11d, dword ptr [rbp+0x500]
       mov      r12d, dword ptr [rbp+0x414]
       mov      r13, qword ptr [rbp+0x3F0]
 
G_M000_IG35:                ;; offset=0x05C9
       add      esi, 16
       cmp      esi, ebx
       jge      G_M000_IG10
 
G_M000_IG36:                ;; offset=0x05D4
       mov      dword ptr [rbp+0x414], r12d
       mov      dword ptr [rbp+0x500], r11d
 
G_M000_IG37:                ;; offset=0x05E2
       xor      ecx, ecx
       cmp      ecx, dword ptr [rbp+0x410]
       jge      SHORT G_M000_IG34
       align    [0 bytes for IG38]
 
G_M000_IG38:                ;; offset=0x05EC
       lea      r9d, [8*rsi]
       movsxd   r9, r9d
       shl      r9, 2
       add      r9, r8
       mov      r11d, ecx
       shl      r11d, 4
       movsxd   r11, r11d
       lea      r9, [r9+4*r11]
       lea      r11d, [8*rbx]
       mov      r12d, dword ptr [rbp+0x414]
       lea      r13d, [r12+rcx]
       mov      eax, r13d
       cdq      
       idiv     edx:eax, dword ptr [rbp+0x41C]
       lea      edx, [rax+rax]
       mov      dword ptr [rbp-0x40], edx
       mov      eax, r13d
       cdq      
       idiv     edx:eax, dword ptr [rbp+0x41C]
       add      edx, edx
       vmovups  zmm0, zmmword ptr [r9]
       lea      eax, [4*r11]
       cdqe     
       vmovups  zmm1, zmmword ptr [r9+4*rax]
       vaddps   zmm0, zmm0, zmm1
       lea      eax, [8*r11]
       cdqe     
       vmovups  zmm2, zmmword ptr [r9+4*rax]
       vaddps   zmm0, zmm0, zmm2
       vsubps   zmm1, zmm1, zmm2
       lea      eax, [r11+2*r11]
       lea      r13d, [4*rax]
       movsxd   r13, r13d
       vsubps   zmm1, zmm1, zmmword ptr [r9+4*r13]
       movsxd   r13, r11d
       vmovups  zmm2, zmmword ptr [r9+4*r13]
       lea      r13d, [r11+4*r11]
       movsxd   rdi, r13d
       vmovups  zmm3, zmmword ptr [r9+4*rdi]
       vaddps   zmm2, zmm2, zmm3
       lea      edi, [r11+8*r11]
       movsxd   rdi, edi
       vmovups  zmm4, zmmword ptr [r9+4*rdi]
       vaddps   zmm2, zmm2, zmm4
       vsubps   zmm3, zmm3, zmm4
       imul     edi, r11d, 13
       movsxd   rdi, edi
       vsubps   zmm3, zmm3, zmmword ptr [r9+4*rdi]
       lea      edi, [r11+r11]
       movsxd   rdi, edi
       vmovups  zmm4, zmmword ptr [r9+4*rdi]
       lea      edi, [rax+rax]
       movsxd   rdi, edi
       vmovups  zmm5, zmmword ptr [r9+4*rdi]
       vaddps   zmm4, zmm4, zmm5
       add      r13d, r13d
       movsxd   rdi, r13d
       vmovups  zmm6, zmmword ptr [r9+4*rdi]
       vaddps   zmm4, zmm6, zmm4
       vsubps   zmm5, zmm5, zmm6
       imul     edi, r11d, 14
       movsxd   rdi, edi
       vsubps   zmm5, zmm5, zmmword ptr [r9+4*rdi]
       movsxd   rdi, eax
       vmovups  zmm6, zmmword ptr [r9+4*rdi]
       lea      edi, [8*r11]
 
G_M000_IG39:                ;; offset=0x072B
       sub      edi, r11d
       movsxd   rdi, edi
       vmovups  zmm7, zmmword ptr [r9+4*rdi]
       vaddps   zmm6, zmm6, zmm7
       imul     edi, r11d, 11
       movsxd   rdi, edi
       vmovups  zmm8, zmmword ptr [r9+4*rdi]
       vaddps   zmm6, zmm6, zmm8
       vsubps   zmm7, zmm7, zmm8
       mov      edi, r11d
       shl      edi, 4
       sub      edi, r11d
       movsxd   rdi, edi
       vsubps   zmm7, zmm7, zmmword ptr [r9+4*rdi]
       vaddps   zmm0, zmm2, zmm0
       vaddps   zmm0, zmm0, zmm4
       vsubps   zmm2, zmm2, zmm4
       vsubps   zmm2, zmm2, zmm6
       vaddps   zmm1, zmm1, zmm3
       vaddps   zmm1, zmm1, zmm5
       vsubps   zmm3, zmm3, zmm5
       vsubps   zmm3, zmm3, zmm7
       mov      edi, dword ptr [rbp-0x40]
       mov      r9d, edi
       imul     r9d, r14d
       add      r9d, edx
       shl      r9d, 4
       movsxd   r9, r9d
       mov      eax, dword ptr [rbp-0x3C]
       mov      r11d, esi
       imul     r11d, eax
       movsxd   r11, r11d
       shl      r11, 2
       mov      r13, qword ptr [rbp+0x3F0]
       add      r11, r13
       lea      r9, [r11+4*r9]
       vmovups  zmmword ptr [r9], zmm0
       lea      r11d, [rdx+0x01]
       cmp      r11d, r14d
       jge      SHORT G_M000_IG41
 
G_M000_IG40:                ;; offset=0x07DD
       vmovups  zmmword ptr [r9+0x40], zmm2
 
G_M000_IG41:                ;; offset=0x07E4
       inc      edi
       mov      r11d, dword ptr [rbp+0x500]
       cmp      edi, r11d
       jge      SHORT G_M000_IG44
 
G_M000_IG42:                ;; offset=0x07F2
       mov      edi, r14d
       shl      edi, 4
       movsxd   rdi, edi
       vmovups  zmmword ptr [r9+4*rdi], zmm1
       inc      edx
       cmp      edx, r14d
       jge      SHORT G_M000_IG44
 
G_M000_IG43:                ;; offset=0x0809
       lea      edi, [r14+0x01]
       shl      edi, 4
       movsxd   rdi, edi
       vmovups  zmmword ptr [r9+4*rdi], zmm3
 
G_M000_IG44:                ;; offset=0x081A
       inc      ecx
       mov      edi, dword ptr [rbp+0x410]
       cmp      ecx, edi
       jge      G_M000_IG35
 
G_M000_IG45:                ;; offset=0x082A
       mov      dword ptr [rbp+0x414], r12d
       mov      dword ptr [rbp+0x500], r11d
       jmp      G_M000_IG38
 
G_M000_IG46:                ;; offset=0x083D
       mov      dword ptr [rbp+0x510], r12d
       jmp      G_M000_IG32
 
G_M000_IG47:                ;; offset=0x0849
       jmp      G_M000_IG16
 
G_M000_IG48:                ;; offset=0x084E
       xor      edi, edi
       jmp      G_M000_IG21
 
G_M000_IG49:                ;; offset=0x0855
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      ecx, 0x7F800000
       andn     edx, edx, ecx
       je       SHORT G_M000_IG48
       inc      edi
       jmp      G_M000_IG20
 
G_M000_IG50:                ;; offset=0x086E
       mov      rsi, rdx
       mov      rdx, r8
       mov      ecx, r15d
       mov      r8d, ebx
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int)]
       mov      r8, qword ptr [rbp+0x3F8]
       jmp      G_M000_IG03
 
G_M000_IG51:                ;; offset=0x088C
       xor      edx, edx
       jmp      G_M000_IG08
 
G_M000_IG52:                ;; offset=0x0893
       movsxd   rsi, edx
       mov      esi, dword ptr [rdi+4*rsi]
       mov      ecx, 0x7F800000
       andn     esi, esi, ecx
       je       SHORT G_M000_IG53
       inc      edx
       mov      r10d, dword ptr [rbp-0x50]
       jmp      G_M000_IG07
 
G_M000_IG53:                ;; offset=0x08B0
       mov      r10d, dword ptr [rbp-0x50]
       jmp      SHORT G_M000_IG51
 
G_M000_IG54:                ;; offset=0x08B6
       mov      r13d, dword ptr [rbp+0x500]
       mov      r9d, dword ptr [rbp+0x414]
       mov      dword ptr [rsp], r9d
       mov      edi, dword ptr [rbp+0x410]
       mov      dword ptr [rsp+0x08], edi
       mov      rdi, r8
       mov      rsi, qword ptr [rbp+0x3F0]
       mov      edx, ebx
       mov      ecx, r13d
       mov      r8d, r14d
       mov      r9d, dword ptr [rbp+0x41C]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
       mov      r11d, r13d
       mov      r12d, dword ptr [rbp+0x414]
       jmp      G_M000_IG10
 
G_M000_IG55:                ;; offset=0x0900
       xor      eax, eax
 
G_M000_IG56:                ;; offset=0x0902
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

; Total bytes of code 2327

