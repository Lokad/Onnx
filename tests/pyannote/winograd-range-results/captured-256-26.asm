; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100
; 44 inlinees with PGO data; 189 single block inlinees; 1 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 440
       lea      rbp, [rsp+0x1E0]
       xor      eax, eax
       mov      qword ptr [rbp-0x168], rax
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x160], xmm8
       mov      rax, -288
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x20], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x10], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x30], rax
       mov      bword ptr [rbp-0x190], rdi
       mov      dword ptr [rbp-0x16C], esi
       mov      bword ptr [rbp-0x198], rdx
       mov      dword ptr [rbp-0x170], ecx
       mov      bword ptr [rbp-0x1A0], r8
       mov      dword ptr [rbp-0x174], r9d
       mov      r15d, dword ptr [rbp+0x60]
       mov      ebx, dword ptr [rbp+0x68]
       mov      eax, dword ptr [rbp+0x70]
       mov      r11d, dword ptr [rbp+0x78]
       mov      r12, bword ptr [rbp+0x30]
       mov      r14d, dword ptr [rbp+0x38]
       mov      r13d, dword ptr [rbp+0x48]
 
G_M000_IG02:                ;; offset=0x0097
       cmp      dword ptr [rbp+0x80], 8
       jne      G_M000_IG97
 
G_M000_IG03:                ;; offset=0x00A4
       cmp      r15d, 16
       jl       G_M000_IG98
       test     r15b, 15
       jne      G_M000_IG98
       cmp      ebx, 32
       jl       G_M000_IG98
       test     bl, 15
       jne      G_M000_IG98
       test     eax, eax
       jle      G_M000_IG98
       test     r11d, r11d
       setle    r8b
       movzx    r8, r8b
 
G_M000_IG04:                ;; offset=0x00DD
       movzx    r8, r8b
       test     r8d, r8d
       jne      G_M000_IG99
       mov      r8d, eax
       add      r8d, 2
       jo       G_M000_IG124
       mov      dword ptr [rbp-0x1A4], r8d
       mov      r8d, r11d
       add      r8d, 2
       jo       G_M000_IG124
       imul     r8d, dword ptr [rbp-0x1A4]
       jo       G_M000_IG124
       imul     r8d, r15d
       jo       G_M000_IG124
       mov      dword ptr [rbp+0x70], eax
       mov      r8d, eax
       add      r8d, 1
       jo       G_M000_IG124
       mov      dword ptr [rbp-0x184], r8d
       sub      r8d, 1
       jo       G_M000_IG124
       imul     r8d, ebx
       jo       G_M000_IG124
       mov      dword ptr [rbp-0x1A4], r8d
       mov      r8d, r11d
       add      r8d, 1
       jo       G_M000_IG124
       sub      r8d, 1
       jo       G_M000_IG124
       imul     r8d, dword ptr [rbp-0x1A4]
       jo       G_M000_IG124
       imul     r8d, dword ptr [rbp+0x80], 2
       jo       G_M000_IG124
       mov      dword ptr [rbp-0x180], r8d
       add      r8d, ebx
       jo       G_M000_IG124
       mov      eax, r8d
       sub      eax, 1
       jo       G_M000_IG124
       mov      r8d, dword ptr [rbp-0x180]
       cdq      
       idiv     edx:eax, r8d
       imul     r8d, eax
       jo       G_M000_IG124
       imul     r8d, r15d
 
G_M000_IG05:                ;; offset=0x01BD
       jo       G_M000_IG124
       imul     r8d, r8d, 9
       jo       G_M000_IG124
       lea      r8, [rbp-0x40]
       mov      qword ptr [rsp], r8
       lea      r8, [rbp-0x30]
       lea      r9, [rbp-0x38]
       mov      edi, r15d
       mov      esi, ebx
       mov      edx, dword ptr [rbp+0x70]
       mov      dword ptr [rbp+0x78], r11d
       mov      ecx, r11d
       call     [Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool]
       test     eax, eax
       je       G_M000_IG122
       mov      eax, dword ptr [rbp+0x70]
       mov      edi, r15d
       imul     edi, eax
       jo       G_M000_IG124
       mov      ecx, dword ptr [rbp+0x78]
       imul     edi, ecx
       jo       G_M000_IG124
       mov      edx, dword ptr [rbp-0x16C]
       cmp      edi, edx
       jne      G_M000_IG100
       imul     edi, r15d, 16
       jo       G_M000_IG124
       imul     edi, ebx
       jo       G_M000_IG124
       mov      esi, dword ptr [rbp-0x170]
       cmp      edi, esi
       jne      G_M000_IG100
       mov      r8d, dword ptr [rbp+0x28]
       cmp      r8d, dword ptr [rbp-0x40]
       jne      G_M000_IG100
       mov      r9d, dword ptr [rbp-0x174]
       test     r9d, r9d
       jne      G_M000_IG54
 
G_M000_IG06:                ;; offset=0x0262
       mov      r10d, dword ptr [rbp+0x18]
       test     r10d, r10d
       jne      G_M000_IG55
 
G_M000_IG07:                ;; offset=0x026F
       cmp      r14d, dword ptr [rbp-0x30]
       jl       G_M000_IG100
       cmp      r13d, dword ptr [rbp-0x38]
       jl       G_M000_IG100
       mov      r11d, dword ptr [rbp+0x58]
       cmp      r11d, dword ptr [rbp-0x40]
       jl       G_M000_IG100
       mov      edi, dword ptr [rbp-0x30]
       cmp      edi, r14d
       ja       G_M000_IG101
       mov      r14d, edi
       mov      edi, dword ptr [rbp-0x38]
       cmp      edi, r13d
       ja       G_M000_IG101
       mov      r13, bword ptr [rbp+0x40]
       mov      r8d, dword ptr [rbp-0x40]
       cmp      r8d, r11d
       mov      dword ptr [rbp-0x174], r9d
       ja       G_M000_IG101
       mov      r11, bword ptr [rbp+0x50]
       mov      r9d, edx
       test     r9d, r9d
       je       SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x02D0
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG09
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp-0x190]
       mov      qword ptr [rbp-0x80], r9
       mov      r9d, edx
       shl      r9, 2
       cmp      qword ptr [rbp-0x80], r9
       jb       G_M000_IG102
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x80]
       jb       G_M000_IG102
 
G_M000_IG09:                ;; offset=0x030B
       mov      r9d, esi
       test     r9d, r9d
       je       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x0313
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG11
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp-0x198]
       mov      qword ptr [rbp-0x88], r9
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0x88], r9
       jb       G_M000_IG102
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x88]
       jb       G_M000_IG102
 
G_M000_IG11:                ;; offset=0x0357
       mov      r9d, dword ptr [rbp-0x174]
       test     r9d, r9d
       je       SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x0363
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG13
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp-0x1A0]
       mov      qword ptr [rbp-0x90], r9
       mov      r9d, dword ptr [rbp-0x174]
       shl      r9, 2
       cmp      qword ptr [rbp-0x90], r9
       jb       G_M000_IG102
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x90]
       jb       G_M000_IG102
 
G_M000_IG13:                ;; offset=0x03AB
       mov      r9d, r10d
       test     r9d, r9d
       je       SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x03B3
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG15
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0x98], r9
       mov      dword ptr [rbp+0x18], r10d
       mov      r9d, r10d
       shl      r9, 2
       cmp      qword ptr [rbp-0x98], r9
       jb       G_M000_IG102
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x98]
       mov      r10d, dword ptr [rbp+0x18]
       jb       G_M000_IG102
 
G_M000_IG15:                ;; offset=0x03FC
       mov      r9d, edx
       test     r9d, r9d
       je       G_M000_IG56
 
G_M000_IG16:                ;; offset=0x0408
       test     r14d, r14d
       mov      dword ptr [rbp+0x18], r10d
       je       SHORT G_M000_IG17
       mov      r10, r12
       sub      r10, qword ptr [rbp-0x190]
       mov      qword ptr [rbp-0xA0], r10
       mov      r10d, edx
       shl      r10, 2
       cmp      qword ptr [rbp-0xA0], r10
       jb       G_M000_IG102
       mov      r10d, r14d
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xA0]
       jb       G_M000_IG102
 
G_M000_IG17:                ;; offset=0x044D
       mov      r10d, esi
       test     r10d, r10d
       je       SHORT G_M000_IG19
 
G_M000_IG18:                ;; offset=0x0455
       test     r14d, r14d
       je       SHORT G_M000_IG19
       mov      r9, r12
       sub      r9, qword ptr [rbp-0x198]
       mov      qword ptr [rbp-0xA8], r9
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0xA8], r9
       jb       G_M000_IG102
       mov      r9d, r14d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xA8]
       jb       G_M000_IG102
 
G_M000_IG19:                ;; offset=0x0496
       mov      r9d, dword ptr [rbp-0x174]
       test     r9d, r9d
       je       SHORT G_M000_IG21
 
G_M000_IG20:                ;; offset=0x04A2
       test     r14d, r14d
       je       SHORT G_M000_IG21
       mov      r10, r12
       sub      r10, qword ptr [rbp-0x1A0]
       mov      qword ptr [rbp-0xB0], r10
       mov      r10d, dword ptr [rbp-0x174]
       mov      r9d, r10d
       shl      r9, 2
       cmp      qword ptr [rbp-0xB0], r9
       jb       G_M000_IG102
       mov      r9d, r14d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xB0]
       mov      dword ptr [rbp-0x174], r10d
       jb       G_M000_IG102
 
G_M000_IG21:                ;; offset=0x04F1
       mov      r9d, dword ptr [rbp+0x18]
       test     r9d, r9d
       je       SHORT G_M000_IG23
 
G_M000_IG22:                ;; offset=0x04FA
       test     r14d, r14d
       je       SHORT G_M000_IG23
       mov      r10, r12
       sub      r10, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xB8], r10
       mov      r10d, dword ptr [rbp+0x18]
       mov      r9d, r10d
       shl      r9, 2
       cmp      qword ptr [rbp-0xB8], r9
       jb       G_M000_IG102
       mov      r9d, r14d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xB8]
       mov      dword ptr [rbp+0x18], r10d
       jb       G_M000_IG102
 
G_M000_IG23:                ;; offset=0x0540
       mov      r9d, edx
       test     r9d, r9d
       je       SHORT G_M000_IG25
 
G_M000_IG24:                ;; offset=0x0548
       test     edi, edi
       je       SHORT G_M000_IG25
       mov      r10, r13
       sub      r10, qword ptr [rbp-0x190]
       mov      qword ptr [rbp-0xC0], r10
       mov      r10d, edx
       shl      r10, 2
       cmp      qword ptr [rbp-0xC0], r10
       jb       G_M000_IG102
       mov      r10d, edi
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xC0]
       jb       G_M000_IG102
 
G_M000_IG25:                ;; offset=0x0588
       mov      r9d, esi
       test     r9d, r9d
       je       SHORT G_M000_IG27
 
G_M000_IG26:                ;; offset=0x0590
       test     edi, edi
       je       SHORT G_M000_IG27
       mov      r9, r13
       sub      r9, qword ptr [rbp-0x198]
       mov      qword ptr [rbp-0xC8], r9
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0xC8], r9
       jb       G_M000_IG102
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xC8]
       jb       G_M000_IG102
 
G_M000_IG27:                ;; offset=0x05D0
       mov      r9d, dword ptr [rbp-0x174]
       test     r9d, r9d
       je       SHORT G_M000_IG29
 
G_M000_IG28:                ;; offset=0x05DC
       test     edi, edi
       je       SHORT G_M000_IG29
       mov      r10, r13
       sub      r10, qword ptr [rbp-0x1A0]
       mov      qword ptr [rbp-0xD0], r10
       mov      r10d, dword ptr [rbp-0x174]
       mov      r9d, r10d
       shl      r9, 2
       cmp      qword ptr [rbp-0xD0], r9
       jb       G_M000_IG102
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xD0]
       mov      dword ptr [rbp-0x174], r10d
       jb       G_M000_IG102
 
G_M000_IG29:                ;; offset=0x062A
       mov      r9d, dword ptr [rbp+0x18]
       test     r9d, r9d
       je       SHORT G_M000_IG31
 
G_M000_IG30:                ;; offset=0x0633
       test     edi, edi
       je       SHORT G_M000_IG31
       mov      r10, r13
       sub      r10, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xD8], r10
       mov      r10d, dword ptr [rbp+0x18]
       mov      r9d, r10d
       shl      r9, 2
       cmp      qword ptr [rbp-0xD8], r9
       jb       G_M000_IG102
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xD8]
       mov      dword ptr [rbp+0x18], r10d
       jb       G_M000_IG102
 
G_M000_IG31:                ;; offset=0x0678
       mov      r9d, edx
       test     r9d, r9d
       je       SHORT G_M000_IG33
 
G_M000_IG32:                ;; offset=0x0680
       test     r8d, r8d
       je       SHORT G_M000_IG33
       mov      r10, r11
       sub      r10, qword ptr [rbp-0x190]
       mov      qword ptr [rbp-0xE0], r10
       mov      r10d, edx
       shl      r10, 2
       cmp      qword ptr [rbp-0xE0], r10
       jb       G_M000_IG102
       mov      r10d, r8d
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xE0]
       jb       G_M000_IG102
 
G_M000_IG33:                ;; offset=0x06C1
       mov      r9d, esi
       test     r9d, r9d
       je       SHORT G_M000_IG35
 
G_M000_IG34:                ;; offset=0x06C9
       test     r8d, r8d
       je       SHORT G_M000_IG35
       mov      r9, r11
       sub      r9, qword ptr [rbp-0x198]
       mov      qword ptr [rbp-0xE8], r9
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0xE8], r9
       jb       G_M000_IG102
       mov      r9d, r8d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xE8]
       jb       G_M000_IG102
 
G_M000_IG35:                ;; offset=0x070A
       mov      r9d, dword ptr [rbp-0x174]
       test     r9d, r9d
       je       SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x0716
       test     r8d, r8d
       je       SHORT G_M000_IG37
       mov      r10, r11
       sub      r10, qword ptr [rbp-0x1A0]
       mov      qword ptr [rbp-0xF0], r10
       mov      r10d, dword ptr [rbp-0x174]
       mov      r9d, r10d
       shl      r9, 2
       cmp      qword ptr [rbp-0xF0], r9
       jb       G_M000_IG102
       mov      r9d, r8d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xF0]
       mov      dword ptr [rbp-0x174], r10d
       jb       G_M000_IG102
 
G_M000_IG37:                ;; offset=0x0765
       mov      r9d, dword ptr [rbp+0x18]
       test     r9d, r9d
       je       SHORT G_M000_IG39
 
G_M000_IG38:                ;; offset=0x076E
       test     r8d, r8d
       je       SHORT G_M000_IG39
       mov      r10, r11
       sub      r10, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xF8], r10
       mov      r10d, dword ptr [rbp+0x18]
       mov      r9d, r10d
       shl      r9, 2
       cmp      qword ptr [rbp-0xF8], r9
       jb       G_M000_IG102
       mov      r9d, r8d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xF8]
       mov      dword ptr [rbp+0x18], r10d
       jb       G_M000_IG102
 
G_M000_IG39:                ;; offset=0x07B4
       mov      r9d, r14d
       test     r9d, r9d
       je       SHORT G_M000_IG41
 
G_M000_IG40:                ;; offset=0x07BC
       test     edi, edi
       je       SHORT G_M000_IG41
       mov      r9, r13
       sub      r9, r12
       mov      qword ptr [rbp-0x100], r9
       mov      r9d, r14d
       shl      r9, 2
       cmp      qword ptr [rbp-0x100], r9
       jb       G_M000_IG102
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x100]
       jb       G_M000_IG102
 
G_M000_IG41:                ;; offset=0x07F8
       mov      r9d, r14d
       test     r9d, r9d
       je       SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x0800
       test     r8d, r8d
       je       SHORT G_M000_IG43
       mov      r9, r11
       sub      r9, r12
       mov      qword ptr [rbp-0x108], r9
       mov      r9d, r14d
       shl      r9, 2
       cmp      qword ptr [rbp-0x108], r9
       jb       G_M000_IG102
       mov      r9d, r8d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x108]
       jb       G_M000_IG102
 
G_M000_IG43:                ;; offset=0x083D
       mov      r9d, r14d
       test     r9d, r9d
       je       SHORT G_M000_IG45
 
G_M000_IG44:                ;; offset=0x0845
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG45
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, r12
       mov      qword ptr [rbp-0x110], r9
       mov      r9d, r14d
       shl      r9, 2
       cmp      qword ptr [rbp-0x110], r9
       jb       G_M000_IG102
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x110]
       jb       G_M000_IG102
 
G_M000_IG45:                ;; offset=0x0885
       mov      r9d, edi
       test     r9d, r9d
       je       SHORT G_M000_IG47
 
G_M000_IG46:                ;; offset=0x088D
       test     r8d, r8d
       je       SHORT G_M000_IG47
       mov      r9, r11
       sub      r9, r13
       mov      qword ptr [rbp-0x118], r9
       mov      r9d, edi
       shl      r9, 2
       cmp      qword ptr [rbp-0x118], r9
       jb       G_M000_IG102
       mov      r9d, r8d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x118]
       jb       G_M000_IG102
 
G_M000_IG47:                ;; offset=0x08CA
       mov      r9d, edi
       test     r9d, r9d
       je       SHORT G_M000_IG49
 
G_M000_IG48:                ;; offset=0x08D2
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG49
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, r13
       mov      qword ptr [rbp-0x120], r9
       mov      dword ptr [rbp+0x48], edi
       mov      r9d, edi
       shl      r9, 2
       cmp      qword ptr [rbp-0x120], r9
       jb       G_M000_IG102
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x120]
       mov      edi, dword ptr [rbp+0x48]
       jb       G_M000_IG102
 
G_M000_IG49:                ;; offset=0x0918
       mov      r9d, r8d
       test     r9d, r9d
       je       SHORT G_M000_IG51
 
G_M000_IG50:                ;; offset=0x0920
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG51
       mov      r9, bword ptr [rbp+0x20]
       mov      bword ptr [rbp+0x50], r11
       sub      r9, r11
       mov      qword ptr [rbp-0x128], r9
       mov      dword ptr [rbp+0x58], r8d
       mov      r9d, r8d
       shl      r9, 2
       cmp      qword ptr [rbp-0x128], r9
       jb       G_M000_IG102
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x128]
       mov      r8d, dword ptr [rbp+0x58]
       mov      r11, bword ptr [rbp+0x50]
       jb       G_M000_IG102
 
G_M000_IG51:                ;; offset=0x0970
       cmp      dword ptr [rbp+0x80], 16
       je       G_M000_IG104
       mov      dword ptr [rbp-0x178], edx
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r9, r9
       cmp      dword ptr [rbp-0x178], 0
       cmovne   r9, bword ptr [rbp-0x190]
       mov      bword ptr [rbp-0x130], r9
       mov      qword ptr [rbp-0x138], r9
       xor      r10d, r10d
       mov      dword ptr [rbp-0x16C], edx
       cmp      edx, 8
       jl       G_M000_IG103
       align    [2 bytes for IG52]
 
G_M000_IG52:                ;; offset=0x09C0
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 7
       add      r9d, r10d
       sar      r9d, 3
       movsxd   r9, r9d
       shl      r9, 5
       mov      rdx, qword ptr [rbp-0x138]
       vpand    ymm1, ymm0, ymmword ptr [r9+rdx]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG105
       add      r10d, 8
       lea      r9d, [r10+0x08]
       mov      edx, dword ptr [rbp-0x16C]
       cmp      r9d, edx
       jg       SHORT G_M000_IG57
 
G_M000_IG53:                ;; offset=0x0A08
       mov      dword ptr [rbp-0x16C], edx
       jmp      SHORT G_M000_IG52
 
G_M000_IG54:                ;; offset=0x0A10
       mov      dword ptr [rbp-0x174], r9d
       cmp      r9d, ebx
       mov      r9d, dword ptr [rbp-0x174]
       jne      G_M000_IG100
       jmp      G_M000_IG06
 
G_M000_IG55:                ;; offset=0x0A2C
       cmp      r10d, dword ptr [rbp-0x40]
       jne      G_M000_IG100
       jmp      G_M000_IG07
 
G_M000_IG56:                ;; offset=0x0A3B
       mov      dword ptr [rbp+0x18], r10d
       jmp      G_M000_IG17
       align    [0 bytes for IG59]
 
G_M000_IG57:                ;; offset=0x0A44
       mov      dword ptr [rbp-0x16C], edx
       cmp      r10d, edx
       jl       G_M000_IG106
       xor      r9d, r9d
       mov      bword ptr [rbp-0x130], r9
       mov      r9d, 1
 
G_M000_IG58:                ;; offset=0x0A63
       xor      r10, r10
       mov      bword ptr [rbp-0x130], r10
       test     r9d, r9d
       je       G_M000_IG122
       mov      r9d, esi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r10, r10
       test     r9d, r9d
       cmovne   r10, bword ptr [rbp-0x198]
       mov      bword ptr [rbp-0x140], r10
       mov      qword ptr [rbp-0x148], r10
       xor      r9d, r9d
       cmp      esi, 8
       jl       G_M000_IG107
       jmp      SHORT G_M000_IG59
 
G_M000_IG59:                ;; offset=0x0AAC
       mov      r10d, r9d
       sar      r10d, 31
       and      r10d, 7
       add      r10d, r9d
       sar      r10d, 3
       movsxd   r10, r10d
       shl      r10, 5
       mov      rdx, qword ptr [rbp-0x148]
       vpand    ymm1, ymm0, ymmword ptr [r10+rdx]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG108
       add      r9d, 8
       lea      r10d, [r9+0x08]
       cmp      r10d, esi
       jle      SHORT G_M000_IG59
 
G_M000_IG60:                ;; offset=0x0AEE
       mov      dword ptr [rbp-0x170], esi
       cmp      r9d, esi
       jl       G_M000_IG109
       xor      edx, edx
       mov      bword ptr [rbp-0x140], rdx
       mov      edx, 1
 
G_M000_IG61:                ;; offset=0x0B0B
       xor      r9, r9
       mov      bword ptr [rbp-0x140], r9
       test     edx, edx
       je       G_M000_IG122
       mov      r10d, dword ptr [rbp-0x174]
       mov      edx, r10d
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r9, r9
       test     edx, edx
       cmovne   r9, bword ptr [rbp-0x1A0]
       mov      bword ptr [rbp-0x150], r9
       xor      edx, edx
       mov      dword ptr [rbp-0x174], r10d
       cmp      r10d, 8
       jl       SHORT G_M000_IG63
       align    [0 bytes for IG62]
 
G_M000_IG62:                ;; offset=0x0B53
       mov      r10d, edx
       sar      r10d, 31
       and      r10d, 7
       add      r10d, edx
       sar      r10d, 3
       movsxd   r10, r10d
       shl      r10, 5
       vpand    ymm1, ymm0, ymmword ptr [r10+r9]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG111
       add      edx, 8
       lea      r10d, [rdx+0x08]
       cmp      r10d, dword ptr [rbp-0x174]
       jle      SHORT G_M000_IG62
 
G_M000_IG63:                ;; offset=0x0B91
       cmp      edx, dword ptr [rbp-0x174]
       jl       G_M000_IG112
       xor      edx, edx
       mov      bword ptr [rbp-0x150], rdx
       mov      edx, 1
 
G_M000_IG64:                ;; offset=0x0BAB
       xor      r9, r9
       mov      bword ptr [rbp-0x150], r9
       test     edx, edx
       je       G_M000_IG122
       mov      r9d, dword ptr [rbp+0x18]
       mov      dword ptr [rbp-0x17C], r9d
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      rdx, rdx
       cmp      dword ptr [rbp-0x17C], 0
       cmovne   rdx, bword ptr [rbp+0x10]
       mov      bword ptr [rbp-0x158], rdx
       xor      r9d, r9d
       cmp      dword ptr [rbp+0x18], 8
       jl       SHORT G_M000_IG66
       align    [1 bytes for IG65]
 
G_M000_IG65:                ;; offset=0x0BF0
       mov      r10d, r9d
       sar      r10d, 31
       and      r10d, 7
       add      r10d, r9d
       sar      r10d, 3
       movsxd   r10, r10d
       shl      r10, 5
       vpand    ymm1, ymm0, ymmword ptr [r10+rdx]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG113
       add      r9d, 8
       lea      r10d, [r9+0x08]
       cmp      r10d, dword ptr [rbp+0x18]
       jle      SHORT G_M000_IG65
 
G_M000_IG66:                ;; offset=0x0C2C
       cmp      r9d, dword ptr [rbp+0x18]
       jl       G_M000_IG114
       xor      edx, edx
       mov      bword ptr [rbp-0x158], rdx
       mov      edx, 1
 
G_M000_IG67:                ;; offset=0x0C44
       xor      r9, r9
       mov      bword ptr [rbp-0x158], r9
       test     edx, edx
       je       G_M000_IG122
       lea      r9d, [rcx+0x01]
       mov      edx, r9d
       shr      edx, 31
       add      r9d, edx
       sar      r9d, 1
       mov      dword ptr [rbp-0x44], r9d
       mov      edx, dword ptr [rbp-0x184]
       shr      edx, 31
       add      edx, dword ptr [rbp-0x184]
       sar      edx, 1
       imul     edx, r9d
       jo       G_M000_IG124
       mov      dword ptr [rbp-0x48], edx
       xor      r10d, r10d
       cmp      r10d, edx
       jl       G_M000_IG84
       jmp      G_M000_IG115
 
G_M000_IG68:                ;; offset=0x0C99
       inc      r10d
       cmp      r10d, 16
       mov      rcx, qword ptr [rbp-0x58]
       mov      r13, bword ptr [rbp+0x40]
       jge      G_M000_IG74
 
G_M000_IG69:                ;; offset=0x0CAE
       xor      r8d, r8d
       cmp      r8d, ebx
       mov      bword ptr [rbp+0x40], r13
       jge      SHORT G_M000_IG68
 
G_M000_IG70:                ;; offset=0x0CBA
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r9d, r10d
       imul     r9d, r15d
       mov      ecx, r9d
       imul     ecx, ebx
       movsxd   rcx, ecx
       shl      rcx, 2
       add      rcx, rsi
       movsxd   r13, r8d
       lea      rcx, [rcx+4*r13]
       shl      r9d, 3
       movsxd   r9, r9d
       lea      r9, [rdi+4*r9]
       test     r15d, r15d
       jle      SHORT G_M000_IG73
 
G_M000_IG71:                ;; offset=0x0D08
       mov      r13d, r15d
       align    [0 bytes for IG72]
 
G_M000_IG72:                ;; offset=0x0D0B
       vmovups  ymm8, ymmword ptr [rcx]
       vbroadcastss ymm9, dword ptr [r9]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r9+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r9+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r9+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r9+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r9+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r9+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r9+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       add      rcx, rax
       add      r9, 32
       dec      r13d
       jne      SHORT G_M000_IG72
 
G_M000_IG73:                ;; offset=0x0D72
       mov      ecx, r10d
       imul     ecx, ebx
       add      ecx, r8d
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
       add      r8d, 8
       cmp      r8d, ebx
       jl       G_M000_IG70
       jmp      G_M000_IG68
 
G_M000_IG74:                ;; offset=0x0DCA
       mov      edi, r11d
       xor      rsi, rsi
       mov      bword ptr [rbp-0x168], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG75:                ;; offset=0x0DDF
       test     edi, edi
       je       SHORT G_M000_IG76
       mov      rsi, r13
 
G_M000_IG76:                ;; offset=0x0DE6
       mov      bword ptr [rbp-0x168], rsi
       xor      edi, edi
       cmp      r11d, 8
       jl       SHORT G_M000_IG78
       align    [0 bytes for IG77]
 
G_M000_IG77:                ;; offset=0x0DF5
       mov      eax, edi
       sar      eax, 31
       and      eax, 7
       add      eax, edi
       sar      eax, 3
       cdqe     
       shl      rax, 5
       vpand    ymm1, ymm0, ymmword ptr [rax+rsi]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG119
       add      edi, 8
       lea      eax, [rdi+0x08]
       cmp      eax, r11d
       jle      SHORT G_M000_IG77
       align    [0 bytes for IG78]
 
G_M000_IG78:                ;; offset=0x0E27
       mov      dword ptr [rbp+0x48], r11d
       cmp      edi, r11d
       jl       G_M000_IG120
       xor      edi, edi
       mov      bword ptr [rbp-0x168], rdi
       mov      edi, 1
 
G_M000_IG79:                ;; offset=0x0E42
       xor      rsi, rsi
       mov      bword ptr [rbp-0x168], rsi
       test     edi, edi
       je       G_M000_IG122
       mov      eax, dword ptr [rbp-0x4C]
       mov      dword ptr [rsp], eax
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
 
G_M000_IG80:                ;; offset=0x0E7F
       mov      bword ptr [rbp-0x68], rdi
 
G_M000_IG81:                ;; offset=0x0E83
       mov      bword ptr [rbp-0x70], rdi
 
G_M000_IG82:                ;; offset=0x0E87
       mov      bword ptr [rbp-0x78], rdi
       mov      eax, dword ptr [rbp-0x4C]
       add      eax, 8
       mov      edi, dword ptr [rbp-0x48]
       cmp      eax, edi
       mov      edx, edi
       mov      r10d, eax
       mov      r9d, dword ptr [rbp-0x44]
       jge      G_M000_IG95
 
G_M000_IG83:                ;; offset=0x0EA5
       mov      eax, dword ptr [rbp+0x70]
       mov      ecx, dword ptr [rbp+0x78]
       mov      edi, dword ptr [rbp+0x48]
       mov      r8d, dword ptr [rbp+0x58]
       mov      r11, bword ptr [rbp+0x50]
 
G_M000_IG84:                ;; offset=0x0EB6
       mov      esi, edx
       sub      esi, r10d
       cmp      esi, 8
       jl       G_M000_IG116
       mov      esi, 8
       mov      dword ptr [rbp+0x48], edi
       mov      dword ptr [rbp+0x58], r8d
       mov      bword ptr [rbp+0x50], r11
 
G_M000_IG85:                ;; offset=0x0ED4
       mov      dword ptr [rbp-0x50], esi
       mov      dword ptr [rbp+0x78], ecx
       mov      dword ptr [rsp], ecx
       mov      dword ptr [rsp+0x08], r9d
       mov      dword ptr [rbp-0x4C], r10d
       mov      dword ptr [rsp+0x10], r10d
       mov      dword ptr [rsp+0x18], esi
       mov      rdi, bword ptr [rbp-0x190]
       mov      esi, dword ptr [rbp-0x16C]
       mov      rdx, r12
       mov      ecx, r14d
       mov      r8d, r15d
       mov      dword ptr [rbp+0x70], eax
       mov      r9d, eax
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      edi, r14d
       xor      rsi, rsi
       mov      bword ptr [rbp-0x160], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG86:                ;; offset=0x0F26
       test     edi, edi
       je       SHORT G_M000_IG87
       mov      rsi, r12
 
G_M000_IG87:                ;; offset=0x0F2D
       mov      bword ptr [rbp-0x160], rsi
       xor      edi, edi
       cmp      r14d, 8
       jl       SHORT G_M000_IG89
       align    [4 bytes for IG88]
 
G_M000_IG88:                ;; offset=0x0F40
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
       jne      G_M000_IG117
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, r14d
       jle      SHORT G_M000_IG88
       align    [0 bytes for IG89]
 
G_M000_IG89:                ;; offset=0x0F73
       cmp      edi, r14d
       jl       G_M000_IG118
       xor      edi, edi
       mov      bword ptr [rbp-0x160], rdi
       mov      edi, 1
 
G_M000_IG90:                ;; offset=0x0F8A
       xor      rsi, rsi
       mov      bword ptr [rbp-0x160], rsi
       test     edi, edi
       je       G_M000_IG122
       xor      rdi, rdi
       test     r14d, r14d
       je       SHORT G_M000_IG91
       mov      rdi, r12
 
G_M000_IG91:                ;; offset=0x0FA5
       mov      bword ptr [rbp-0x60], rdi
       xor      rsi, rsi
       cmp      dword ptr [rbp-0x170], 0
       je       SHORT G_M000_IG92
       mov      r10, bword ptr [rbp-0x198]
       mov      rsi, r10
 
G_M000_IG92:                ;; offset=0x0FBE
       mov      bword ptr [rbp-0x68], rsi
       xor      rdx, rdx
       mov      r11d, dword ptr [rbp+0x48]
       test     r11d, r11d
       je       SHORT G_M000_IG93
       mov      bword ptr [rbp+0x40], r13
       mov      rdx, r13
       mov      r13, bword ptr [rbp+0x40]
 
G_M000_IG93:                ;; offset=0x0FD8
       mov      bword ptr [rbp-0x70], rdx
       xor      rcx, rcx
       cmp      dword ptr [rbp+0x58], 0
       je       SHORT G_M000_IG94
       mov      r8, bword ptr [rbp+0x50]
       mov      rcx, r8
 
G_M000_IG94:                ;; offset=0x0FEB
       mov      bword ptr [rbp-0x78], rcx
       mov      qword ptr [rbp-0x58], rcx
       xor      r10d, r10d
       movsxd   rax, ebx
       shl      rax, 2
       mov      rcx, qword ptr [rbp-0x58]
       jmp      G_M000_IG69
 
G_M000_IG95:                ;; offset=0x1006
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG122
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG122
       mov      rdi, bword ptr [rbp-0x1A0]
       mov      esi, dword ptr [rbp-0x174]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG122
       mov      rdi, bword ptr [rbp+0x10]
       mov      esi, dword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG122
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
       mov      r8, bword ptr [rbp-0x1A0]
       mov      r9d, dword ptr [rbp-0x174]
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG96:                ;; offset=0x10BC
       vzeroupper 
       add      rsp, 440
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG97:                ;; offset=0x10D1
       cmp      dword ptr [rbp+0x80], 16
       je       G_M000_IG03
 
G_M000_IG98:                ;; offset=0x10DE
       mov      r8d, 1
       jmp      G_M000_IG04
 
G_M000_IG99:                ;; offset=0x10E9
       mov      rdi, 0x761AD707CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0xD4C
       mov      rsi, 0x761AD7131A30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG100:                ;; offset=0x1125
       mov      rdi, 0x761AD707CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0xCE4
       mov      rsi, 0x761AD7131A30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG101:                ;; offset=0x1161
       call     [System.ThrowHelper:ThrowArgumentOutOfRangeException()]
       int3     
 
G_M000_IG102:                ;; offset=0x1168
       mov      rdi, 0x761AD707CD28
       call     CORINFO_HELP_NEWSFAST
       mov      r12, rax
       mov      edi, 0xD12
       mov      rsi, 0x761AD7131A30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, r12
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, r12
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG103:                ;; offset=0x11A4
       mov      edx, dword ptr [rbp-0x16C]
       jmp      G_M000_IG57
 
G_M000_IG104:                ;; offset=0x11AF
       mov      rdi, 0x761AD797DBD8
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      rdi, rbx
       call     [System.PlatformNotSupportedException:.ctor():this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG105:                ;; offset=0x11D3
       xor      r9d, r9d
       jmp      G_M000_IG58
 
G_M000_IG106:                ;; offset=0x11DB
       movsxd   r9, r10d
       mov      rdx, qword ptr [rbp-0x138]
       mov      r9d, dword ptr [rdx+4*r9]
       mov      dword ptr [rbp-0x1A4], r9d
       mov      r9d, 0x7F800000
       mov      dword ptr [rbp-0x1A8], r9d
       mov      r9d, dword ptr [rbp-0x1A4]
       andn     r9d, r9d, dword ptr [rbp-0x1A8]
       je       SHORT G_M000_IG105
       inc      r10d
       mov      edx, dword ptr [rbp-0x16C]
       jmp      G_M000_IG57
 
G_M000_IG107:                ;; offset=0x121D
       mov      rdx, qword ptr [rbp-0x148]
       jmp      G_M000_IG60
 
G_M000_IG108:                ;; offset=0x1229
       xor      edx, edx
       mov      dword ptr [rbp-0x170], esi
       jmp      G_M000_IG61
 
G_M000_IG109:                ;; offset=0x1236
       movsxd   r10, r9d
       mov      r10d, dword ptr [rdx+4*r10]
       mov      dword ptr [rbp-0x1A8], r10d
       mov      r10d, 0x7F800000
       mov      dword ptr [rbp-0x1A4], r10d
       mov      r10d, dword ptr [rbp-0x1A8]
       andn     r10d, r10d, dword ptr [rbp-0x1A4]
       je       SHORT G_M000_IG110
       inc      r9d
       mov      esi, dword ptr [rbp-0x170]
       jmp      G_M000_IG60
 
G_M000_IG110:                ;; offset=0x1271
       mov      esi, dword ptr [rbp-0x170]
       jmp      SHORT G_M000_IG108
 
G_M000_IG111:                ;; offset=0x1279
       xor      edx, edx
       jmp      G_M000_IG64
 
G_M000_IG112:                ;; offset=0x1280
       movsxd   r10, edx
       mov      r10d, dword ptr [r9+4*r10]
       mov      dword ptr [rbp-0x1A4], r10d
       mov      r10d, 0x7F800000
       mov      dword ptr [rbp-0x1A8], r10d
       mov      r10d, dword ptr [rbp-0x1A4]
       andn     r10d, r10d, dword ptr [rbp-0x1A8]
       je       SHORT G_M000_IG111
       inc      edx
       jmp      G_M000_IG63
 
G_M000_IG113:                ;; offset=0x12B4
       xor      edx, edx
       jmp      G_M000_IG67
 
G_M000_IG114:                ;; offset=0x12BB
       movsxd   r10, r9d
       mov      r10d, dword ptr [rdx+4*r10]
       mov      dword ptr [rbp-0x1A8], r10d
       mov      r10d, 0x7F800000
       mov      dword ptr [rbp-0x1A4], r10d
       mov      r10d, dword ptr [rbp-0x1A8]
       andn     r10d, r10d, dword ptr [rbp-0x1A4]
       je       SHORT G_M000_IG113
       inc      r9d
       jmp      G_M000_IG66
 
G_M000_IG115:                ;; offset=0x12F0
       mov      dword ptr [rbp+0x78], ecx
       mov      dword ptr [rbp+0x70], eax
       mov      dword ptr [rbp+0x58], r8d
       mov      bword ptr [rbp+0x50], r11
       jmp      G_M000_IG95
 
G_M000_IG116:                ;; offset=0x1303
       mov      dword ptr [rbp+0x48], edi
       mov      dword ptr [rbp+0x58], r8d
       mov      bword ptr [rbp+0x50], r11
       jmp      G_M000_IG85
 
G_M000_IG117:                ;; offset=0x1313
       xor      edi, edi
       jmp      G_M000_IG90
 
G_M000_IG118:                ;; offset=0x131A
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      ecx, 0x7F800000
       andn     edx, edx, ecx
       je       SHORT G_M000_IG117
       inc      edi
       jmp      G_M000_IG89
 
G_M000_IG119:                ;; offset=0x1333
       xor      edi, edi
       mov      dword ptr [rbp+0x48], r11d
       jmp      G_M000_IG79
 
G_M000_IG120:                ;; offset=0x133E
       movsxd   rax, edi
       mov      eax, dword ptr [rsi+4*rax]
       mov      r10d, 0x7F800000
       andn     eax, eax, r10d
       je       SHORT G_M000_IG121
       inc      edi
       mov      r11d, dword ptr [rbp+0x48]
       jmp      G_M000_IG78
 
G_M000_IG121:                ;; offset=0x135C
       mov      r11d, dword ptr [rbp+0x48]
       jmp      SHORT G_M000_IG119
 
G_M000_IG122:                ;; offset=0x1362
       xor      eax, eax
 
G_M000_IG123:                ;; offset=0x1364
       vzeroupper 
       add      rsp, 440
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG124:                ;; offset=0x1379
       call     CORINFO_HELP_OVERFLOW
       int3     
 
RWD00  	dd	7F800000h		;       inf

; Total bytes of code 4991

