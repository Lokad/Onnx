; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x3af
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 5
; 7 inlinees with PGO data; 220 single block inlinees; 52 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 192
       mov      qword ptr [rsp+0x558], r15
       mov      qword ptr [rsp+0x550], r14
       mov      qword ptr [rsp+0x548], r13
       mov      qword ptr [rsp+0x540], r12
       mov      qword ptr [rsp+0x538], rbx
       lea      rbp, [rsp+0xC0]
       xor      eax, eax
       mov      qword ptr [rbp-0x30], rax
       mov      r15d, dword ptr [rbp+0x500]
       mov      ebx, dword ptr [rbp+0x508]
       mov      r14d, dword ptr [rbp+0x510]
       mov      r13d, dword ptr [rbp+0x518]
       mov      r12d, dword ptr [rbp+0x520]
       mov      r11d, dword ptr [rbp+0x42C]
       mov      r10d, dword ptr [rbp+0x428]
       mov      eax, dword ptr [rbp+0x424]
 
G_M000_IG02:                ;; offset=0x0078
       mov      r9, bword ptr [rbp+0x458]
       mov      bword ptr [rbp-0x58], r9
       mov      r8d, dword ptr [rbp+0x460]
       mov      dword ptr [rbp-0x34], r8d
       mov      rcx, bword ptr [rbp+0x448]
       mov      bword ptr [rbp-0x60], rcx
       mov      edx, dword ptr [rbp+0x450]
       mov      dword ptr [rbp-0x38], edx
       mov      rsi, bword ptr [rbp+0x4D0]
       mov      bword ptr [rbp-0x68], rsi
       mov      edi, dword ptr [rbp+0x4D8]
       mov      dword ptr [rbp-0x3C], edi
       mov      rcx, bword ptr [rbp+0x4E0]
       mov      bword ptr [rbp-0x70], rcx
       mov      edx, dword ptr [rbp+0x4E8]
       mov      dword ptr [rbp-0x40], edx
       mov      r9, bword ptr [rbp+0x4F0]
       mov      bword ptr [rbp-0x78], r9
       mov      r9d, dword ptr [rbp+0x4F8]
       mov      dword ptr [rbp-0x44], r9d
       mov      r8, bword ptr [rbp+0x468]
       mov      bword ptr [rbp-0x80], r8
       mov      r8d, dword ptr [rbp+0x470]
       mov      dword ptr [rbp-0x48], r8d
       mov      r8, bword ptr [rbp+0x4B0]
       mov      bword ptr [rbp-0x88], r8
       mov      r8d, dword ptr [rbp+0x4B8]
       mov      dword ptr [rbp-0x4C], r8d
       mov      r8, bword ptr [rbp+0x4C0]
       mov      bword ptr [rbp-0x90], r8
       mov      r8d, dword ptr [rbp+0x4C8]
       mov      dword ptr [rbp-0x50], r8d
       cmp      eax, r10d
       jl       G_M000_IG32
 
G_M000_IG03:                ;; offset=0x0131
       mov      rdi, bword ptr [rbp-0x78]
       mov      esi, dword ptr [rbp-0x44]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG16
 
G_M000_IG04:                ;; offset=0x0146
       mov      rdi, bword ptr [rbp-0x78]
       mov      esi, dword ptr [rbp-0x44]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG16
 
G_M000_IG05:                ;; offset=0x015B
       mov      rdi, bword ptr [rbp-0x60]
       mov      esi, dword ptr [rbp-0x38]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG16
 
G_M000_IG06:                ;; offset=0x0170
       mov      rdi, bword ptr [rbp-0x88]
       mov      esi, dword ptr [rbp-0x4C]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG16
 
G_M000_IG07:                ;; offset=0x0188
       lea      rdi, [rsp]
       lea      rsi, [rbp+0x4B0]
       mov      rcx, bword ptr [rsi]
       mov      bword ptr [rsp], rcx
       add      rsi, 8
       add      rdi, 8
       movsq    
       mov      dword ptr [rsp+0x10], ebx
       imul     r14d, r13d
       mov      dword ptr [rsp+0x18], r14d
       mov      dword ptr [rsp+0x20], r12d
       movzx    r8, byte  ptr [rbp+0x528]
       mov      dword ptr [rsp+0x28], r8d
       mov      r8, bword ptr [rbp-0x60]
       mov      r9d, dword ptr [rbp-0x38]
       mov      rdx, bword ptr [rbp-0x90]
       mov      ecx, dword ptr [rbp-0x50]
       mov      rdi, bword ptr [rbp-0x78]
       mov      esi, dword ptr [rbp-0x44]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG08:                ;; offset=0x01E7
       vzeroupper 
       add      rsp, 0x538
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG09:                ;; offset=0x01FC
       xor      edx, edx
       jmp      SHORT G_M000_IG12
       align    [0 bytes for IG13]
 
G_M000_IG10:                ;; offset=0x0200
       inc      edx
       cmp      edx, 16
       jge      G_M000_IG25
 
G_M000_IG11:                ;; offset=0x020B
       mov      r9, qword ptr [rbp+0x408]
 
G_M000_IG12:                ;; offset=0x0212
       xor      r11d, r11d
       cmp      r11d, ebx
       jge      SHORT G_M000_IG10
 
G_M000_IG13:                ;; offset=0x021A
       mov      r8d, edx
       imul     r8d, r15d
       mov      r12d, r8d
       imul     r12d, ebx
       movsxd   r12, r12d
       shl      r12, 2
       add      r12, rsi
       movsxd   r10, r11d
       lea      r12, [r12+4*r10]
       shl      r8d, 3
       movsxd   r8, r8d
       lea      r8, [rdi+4*r8]
       test     r15d, r15d
       jg       G_M000_IG53
       mov      r8d, edx
       imul     r8d, ebx
       add      r8d, r11d
       shl      r8d, 3
       movsxd   r8, r8d
       lea      r8, [r9+4*r8]
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [r8], ymm0
       vmovups  ymmword ptr [r8+0x20], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [r8+0x40], ymm0
       vmovups  ymmword ptr [r8+0x60], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [r8+0x80], ymm0
       vmovups  ymmword ptr [r8+0xA0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [r8+0xC0], ymm0
       vmovups  ymmword ptr [r8+0xE0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [r8+0x100], ymm0
       vmovups  ymmword ptr [r8+0x120], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [r8+0x140], ymm0
       vmovups  ymmword ptr [r8+0x160], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [r8+0x180], ymm0
       vmovups  ymmword ptr [r8+0x1A0], ymm0
       vxorps   ymm0, ymm0, ymm0
       vmovups  ymmword ptr [r8+0x1C0], ymm0
       vmovups  ymmword ptr [r8+0x1E0], ymm0
       add      r11d, 16
       cmp      r11d, ebx
       jge      G_M000_IG10
 
G_M000_IG14:                ;; offset=0x0312
       mov      r9, qword ptr [rbp+0x408]
       jmp      G_M000_IG13
 
G_M000_IG15:                ;; offset=0x031E
       mov      eax, dword ptr [rbp+0x424]
       mov      dword ptr [rsp], eax
       mov      edi, dword ptr [rbp+0x420]
       mov      dword ptr [rsp+0x08], edi
       mov      rdi, qword ptr [rbp+0x408]
       mov      rsi, qword ptr [rbp+0x400]
       mov      edx, ebx
       mov      ecx, r14d
       mov      r8d, r13d
       mov      r9d, dword ptr [rbp+0x42C]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd512(ptr,ptr,int,int,int,int,int,int)]
       jmp      G_M000_IG28
       align    [0 bytes for IG21]
 
G_M000_IG16:                ;; offset=0x0359
       xor      eax, eax
 
G_M000_IG17:                ;; offset=0x035B
       vzeroupper 
       add      rsp, 0x538
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG18:                ;; offset=0x0370
       jmp      G_M000_IG34
 
G_M000_IG19:                ;; offset=0x0375
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      ecx, edx
       imul     ecx, r15d
       mov      r8d, ecx
       imul     r8d, ebx
       movsxd   r8, r8d
       shl      r8, 2
       add      r8, rsi
       movsxd   r12, r10d
       lea      r8, [r8+4*r12]
       shl      ecx, 3
       movsxd   rcx, ecx
       lea      rcx, [rdi+4*rcx]
       test     r15d, r15d
       jle      SHORT G_M000_IG22
 
G_M000_IG20:                ;; offset=0x03C2
       mov      r12d, r15d
 
G_M000_IG21:                ;; offset=0x03C5
       vmovups  ymm8, ymmword ptr [r8]
       vbroadcastss ymm9, dword ptr [rcx]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rcx+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rcx+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rcx+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rcx+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rcx+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rcx+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [rcx+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       add      r8, r11
       add      rcx, 32
       dec      r12d
       jne      SHORT G_M000_IG21
 
G_M000_IG22:                ;; offset=0x042D
       mov      ecx, edx
       imul     ecx, ebx
       add      ecx, r10d
       shl      ecx, 3
       movsxd   rcx, ecx
       lea      rcx, [r9+4*rcx]
       vmovups  ymmword ptr [rcx], ymm0
       vmovups  ymmword ptr [rcx+0x20], ymm1
       vmovups  ymmword ptr [rcx+0x40], ymm2
       vmovups  ymmword ptr [rcx+0x60], ymm3
       vmovups  ymmword ptr [rcx+0x80], ymm4
       vmovups  ymmword ptr [rcx+0xA0], ymm5
       vmovups  ymmword ptr [rcx+0xC0], ymm6
       vmovups  ymmword ptr [rcx+0xE0], ymm7
       add      r10d, 8
       cmp      r10d, ebx
       jl       G_M000_IG19
 
G_M000_IG23:                ;; offset=0x047F
       inc      edx
       cmp      edx, 16
       jge      SHORT G_M000_IG25
 
G_M000_IG24:                ;; offset=0x0486
       xor      r10d, r10d
       cmp      r10d, ebx
       jl       G_M000_IG19
       jmp      SHORT G_M000_IG23
 
G_M000_IG25:                ;; offset=0x0494
       mov      rdi, bword ptr [rbp-0x70]
       mov      esi, dword ptr [rbp-0x40]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG16
 
G_M000_IG26:                ;; offset=0x04A9
       mov      r12d, dword ptr [rbp+0x520]
       cmp      r12d, 16
       je       G_M000_IG15
 
G_M000_IG27:                ;; offset=0x04BA
       mov      eax, dword ptr [rbp+0x424]
       mov      dword ptr [rsp], eax
       mov      r10d, dword ptr [rbp+0x420]
       mov      dword ptr [rsp+0x08], r10d
       mov      rdi, qword ptr [rbp+0x408]
       mov      rsi, qword ptr [rbp+0x400]
       mov      edx, ebx
       mov      ecx, r14d
       mov      r8d, r13d
       mov      r9d, dword ptr [rbp+0x42C]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
 
G_M000_IG28:                ;; offset=0x04F2
       xor      eax, eax
       mov      bword ptr [rbp+0x3F8], rax
 
G_M000_IG29:                ;; offset=0x04FB
       mov      bword ptr [rbp+0x3F0], rax
 
G_M000_IG30:                ;; offset=0x0502
       mov      bword ptr [rbp+0x3E8], rax
 
G_M000_IG31:                ;; offset=0x0509
       mov      bword ptr [rbp+0x3E0], rax
       mov      eax, dword ptr [rbp+0x424]
       add      eax, 8
       mov      r10d, dword ptr [rbp+0x428]
       cmp      eax, r10d
       mov      r11d, dword ptr [rbp+0x42C]
       jge      G_M000_IG03
 
G_M000_IG32:                ;; offset=0x0530
       mov      dword ptr [rbp+0x428], r10d
       mov      r8d, r10d
       sub      r8d, eax
       cmp      r8d, 8
       jl       G_M000_IG18
 
G_M000_IG33:                ;; offset=0x0547
       mov      r8d, 8
 
G_M000_IG34:                ;; offset=0x054D
       mov      dword ptr [rbp+0x420], r8d
       mov      dword ptr [rsp], r13d
       mov      dword ptr [rbp+0x42C], r11d
       mov      dword ptr [rsp+0x08], r11d
       mov      dword ptr [rbp+0x424], eax
       mov      dword ptr [rsp+0x10], eax
       mov      dword ptr [rsp+0x18], r8d
       mov      rdi, bword ptr [rbp-0x80]
       mov      esi, dword ptr [rbp-0x48]
       mov      rdx, bword ptr [rbp-0x68]
       mov      ecx, dword ptr [rbp-0x3C]
       mov      r8d, r15d
       mov      r9d, r14d
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      eax, dword ptr [rbp-0x3C]
       mov      edi, eax
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG35:                ;; offset=0x05A1
       test     edi, edi
       je       SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x05A5
       mov      rcx, bword ptr [rbp-0x68]
       mov      rsi, rcx
 
G_M000_IG37:                ;; offset=0x05AC
       mov      bword ptr [rbp-0x30], rsi
       xor      edi, edi
       cmp      eax, 8
       jl       SHORT G_M000_IG39
       align    [0 bytes for IG38]
 
G_M000_IG38:                ;; offset=0x05B7
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
       jne      G_M000_IG51
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, eax
       jle      SHORT G_M000_IG38
       align    [0 bytes for IG39]
 
G_M000_IG39:                ;; offset=0x05E9
       cmp      edi, eax
       jl       G_M000_IG52
       xor      edi, edi
       mov      bword ptr [rbp-0x30], rdi
       mov      edi, 1
 
G_M000_IG40:                ;; offset=0x05FC
       xor      rsi, rsi
       mov      bword ptr [rbp-0x30], rsi
       test     edi, edi
       je       G_M000_IG16
 
G_M000_IG41:                ;; offset=0x060A
       xor      rdi, rdi
       test     eax, eax
       je       SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x0610
       mov      rcx, bword ptr [rbp-0x68]
       mov      rdi, rcx
 
G_M000_IG43:                ;; offset=0x0617
       mov      bword ptr [rbp+0x3F8], rdi
       xor      rsi, rsi
       cmp      dword ptr [rbp-0x34], 0
       je       SHORT G_M000_IG45
 
G_M000_IG44:                ;; offset=0x0626
       mov      r8, bword ptr [rbp-0x58]
       mov      rsi, r8
 
G_M000_IG45:                ;; offset=0x062D
       mov      bword ptr [rbp+0x3F0], rsi
       xor      r9, r9
       cmp      dword ptr [rbp-0x40], 0
       je       SHORT G_M000_IG47
 
G_M000_IG46:                ;; offset=0x063D
       mov      r11, bword ptr [rbp-0x70]
       mov      r9, r11
 
G_M000_IG47:                ;; offset=0x0644
       mov      bword ptr [rbp+0x3E8], r9
       mov      qword ptr [rbp+0x408], r9
       xor      r8, r8
       cmp      dword ptr [rbp-0x44], 0
       je       SHORT G_M000_IG49
 
G_M000_IG48:                ;; offset=0x065B
       mov      r8, bword ptr [rbp-0x78]
 
G_M000_IG49:                ;; offset=0x065F
       mov      bword ptr [rbp+0x3E0], r8
       mov      qword ptr [rbp+0x400], r8
       mov      dword ptr [rbp+0x520], r12d
       cmp      r12d, 16
       je       G_M000_IG09
 
G_M000_IG50:                ;; offset=0x067E
       xor      edx, edx
       movsxd   r11, ebx
       shl      r11, 2
       jmp      G_M000_IG24
 
G_M000_IG51:                ;; offset=0x068C
       xor      edi, edi
       jmp      G_M000_IG40
 
G_M000_IG52:                ;; offset=0x0693
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      r8d, 0x7F800000
       andn     edx, edx, r8d
       je       SHORT G_M000_IG51
       inc      edi
       jmp      G_M000_IG39
 
G_M000_IG53:                ;; offset=0x06AD
       cmp      byte  ptr [r12], r12b
       cmp      dword ptr [r8], r8d
       call     CORINFO_HELP_THROW_PLATFORM_NOT_SUPPORTED
       int3     
 
RWD00  	dd	7F800000h		;       inf

; Total bytes of code 1722

