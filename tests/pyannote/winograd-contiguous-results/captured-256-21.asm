; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100
; 42 inlinees with PGO data; 220 single block inlinees; 17 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     r12
       push     rbx
       sub      rsp, 440
       lea      rbp, [rsp+0x1E0]
       vxorps   xmm8, xmm8, xmm8
       vmovdqa  xmmword ptr [rbp-0x170], xmm8
       vmovdqa  xmmword ptr [rbp-0x160], xmm8
       mov      rax, -288
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x20], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x10], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      qword ptr [rbp-0x30], rax
       mov      bword ptr [rbp-0x198], rdi
       mov      dword ptr [rbp-0x174], esi
       mov      bword ptr [rbp-0x1A0], rdx
       mov      dword ptr [rbp-0x178], ecx
       mov      bword ptr [rbp-0x1A8], r8
       mov      dword ptr [rbp-0x17C], r9d
       mov      r15d, dword ptr [rbp+0x60]
       mov      ebx, dword ptr [rbp+0x68]
       mov      eax, dword ptr [rbp+0x70]
       mov      r11d, dword ptr [rbp+0x78]
       mov      r13, bword ptr [rbp+0x30]
       mov      r14d, dword ptr [rbp+0x38]
       mov      r12d, dword ptr [rbp+0x48]
 
G_M000_IG02:                ;; offset=0x0096
       cmp      dword ptr [rbp+0x80], 8
       jne      G_M000_IG96
 
G_M000_IG03:                ;; offset=0x00A3
       cmp      r15d, 16
       jl       G_M000_IG97
       test     r15b, 15
       jne      G_M000_IG97
       cmp      ebx, 32
       jl       G_M000_IG97
       test     bl, 15
       jne      G_M000_IG97
       test     eax, eax
       jle      G_M000_IG97
       test     r11d, r11d
       setle    r8b
       movzx    r8, r8b
 
G_M000_IG04:                ;; offset=0x00DC
       movzx    r8, r8b
       test     r8d, r8d
       jne      G_M000_IG98
       mov      r8d, eax
       add      r8d, 2
       jo       G_M000_IG121
       mov      dword ptr [rbp-0x1AC], r8d
       mov      r8d, r11d
       add      r8d, 2
       jo       G_M000_IG121
       imul     r8d, dword ptr [rbp-0x1AC]
       jo       G_M000_IG121
       imul     r8d, r15d
       jo       G_M000_IG121
       mov      dword ptr [rbp+0x70], eax
       mov      r8d, eax
       add      r8d, 1
       jo       G_M000_IG121
       mov      dword ptr [rbp-0x190], r8d
       sub      r8d, 1
       jo       G_M000_IG121
       imul     r8d, ebx
       jo       G_M000_IG121
       mov      dword ptr [rbp-0x1AC], r8d
       mov      r8d, r11d
       add      r8d, 1
       jo       G_M000_IG121
       sub      r8d, 1
       jo       G_M000_IG121
       imul     r8d, dword ptr [rbp-0x1AC]
       jo       G_M000_IG121
       imul     r8d, dword ptr [rbp+0x80], 2
       jo       G_M000_IG121
       mov      dword ptr [rbp-0x18C], r8d
       add      r8d, ebx
       jo       G_M000_IG121
       mov      eax, r8d
       sub      eax, 1
       jo       G_M000_IG121
       mov      r8d, dword ptr [rbp-0x18C]
       cdq      
       idiv     edx:eax, r8d
       imul     r8d, eax
       jo       G_M000_IG121
       imul     r8d, r15d
 
G_M000_IG05:                ;; offset=0x01BC
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
       mov      dword ptr [rbp+0x78], r11d
       mov      ecx, r11d
       call     [Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool]
       test     eax, eax
       je       G_M000_IG119
       mov      eax, dword ptr [rbp+0x70]
       mov      edi, r15d
       imul     edi, eax
       jo       G_M000_IG121
       mov      ecx, dword ptr [rbp+0x78]
       imul     edi, ecx
       jo       G_M000_IG121
       mov      edx, dword ptr [rbp-0x174]
       cmp      edi, edx
       jne      G_M000_IG99
       imul     edi, r15d, 16
       jo       G_M000_IG121
       imul     edi, ebx
       jo       G_M000_IG121
       mov      esi, dword ptr [rbp-0x178]
       cmp      edi, esi
       jne      G_M000_IG99
       mov      r8d, dword ptr [rbp+0x28]
       cmp      r8d, dword ptr [rbp-0x40]
       jne      G_M000_IG99
       mov      r9d, dword ptr [rbp-0x17C]
       test     r9d, r9d
       jne      G_M000_IG54
 
G_M000_IG06:                ;; offset=0x0261
       mov      r10d, dword ptr [rbp+0x18]
       test     r10d, r10d
       jne      G_M000_IG55
 
G_M000_IG07:                ;; offset=0x026E
       cmp      r14d, dword ptr [rbp-0x30]
       jl       G_M000_IG99
       cmp      r12d, dword ptr [rbp-0x38]
       jl       G_M000_IG99
       mov      r11d, dword ptr [rbp+0x58]
       cmp      r11d, dword ptr [rbp-0x40]
       jl       G_M000_IG99
       mov      edi, dword ptr [rbp-0x30]
       cmp      edi, r14d
       ja       G_M000_IG100
       mov      r14d, edi
       mov      edi, dword ptr [rbp-0x38]
       cmp      edi, r12d
       ja       G_M000_IG100
       mov      r12, bword ptr [rbp+0x40]
       mov      r8d, dword ptr [rbp-0x40]
       cmp      r8d, r11d
       mov      dword ptr [rbp-0x17C], r9d
       ja       G_M000_IG100
       mov      r11, bword ptr [rbp+0x50]
       mov      r9d, edx
       test     r9d, r9d
       je       SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x02CF
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG09
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp-0x198]
       mov      qword ptr [rbp-0x88], r9
       mov      r9d, edx
       shl      r9, 2
       cmp      qword ptr [rbp-0x88], r9
       jb       G_M000_IG101
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x88]
       jb       G_M000_IG101
 
G_M000_IG09:                ;; offset=0x0313
       mov      r9d, esi
       test     r9d, r9d
       je       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x031B
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG11
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp-0x1A0]
       mov      qword ptr [rbp-0x90], r9
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0x90], r9
       jb       G_M000_IG101
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x90]
       jb       G_M000_IG101
 
G_M000_IG11:                ;; offset=0x035F
       mov      r9d, dword ptr [rbp-0x17C]
       test     r9d, r9d
       je       SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x036B
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG13
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp-0x1A8]
       mov      qword ptr [rbp-0x98], r9
       mov      r9d, dword ptr [rbp-0x17C]
       shl      r9, 2
       cmp      qword ptr [rbp-0x98], r9
       jb       G_M000_IG101
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x98]
       jb       G_M000_IG101
 
G_M000_IG13:                ;; offset=0x03B3
       mov      r9d, r10d
       test     r9d, r9d
       je       SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x03BB
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG15
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xA0], r9
       mov      dword ptr [rbp+0x18], r10d
       mov      r9d, r10d
       shl      r9, 2
       cmp      qword ptr [rbp-0xA0], r9
       jb       G_M000_IG101
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xA0]
       mov      r10d, dword ptr [rbp+0x18]
       jb       G_M000_IG101
 
G_M000_IG15:                ;; offset=0x0404
       mov      r9d, edx
       test     r9d, r9d
       je       G_M000_IG56
 
G_M000_IG16:                ;; offset=0x0410
       test     r14d, r14d
       mov      dword ptr [rbp+0x18], r10d
       je       SHORT G_M000_IG17
       mov      r10, r13
       sub      r10, qword ptr [rbp-0x198]
       mov      qword ptr [rbp-0xA8], r10
       mov      r10d, edx
       shl      r10, 2
       cmp      qword ptr [rbp-0xA8], r10
       jb       G_M000_IG101
       mov      r10d, r14d
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xA8]
       jb       G_M000_IG101
 
G_M000_IG17:                ;; offset=0x0455
       mov      r10d, esi
       test     r10d, r10d
       je       SHORT G_M000_IG19
 
G_M000_IG18:                ;; offset=0x045D
       test     r14d, r14d
       je       SHORT G_M000_IG19
       mov      r9, r13
       sub      r9, qword ptr [rbp-0x1A0]
       mov      qword ptr [rbp-0xB0], r9
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0xB0], r9
       jb       G_M000_IG101
       mov      r9d, r14d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xB0]
       jb       G_M000_IG101
 
G_M000_IG19:                ;; offset=0x049E
       mov      r10d, dword ptr [rbp-0x17C]
       test     r10d, r10d
       je       SHORT G_M000_IG21
 
G_M000_IG20:                ;; offset=0x04AA
       test     r14d, r14d
       je       SHORT G_M000_IG21
       mov      r9, r13
       sub      r9, qword ptr [rbp-0x1A8]
       mov      qword ptr [rbp-0xB8], r9
       mov      r9d, dword ptr [rbp-0x17C]
       mov      r10d, r9d
       shl      r10, 2
       cmp      qword ptr [rbp-0xB8], r10
       jb       G_M000_IG101
       mov      r10d, r14d
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xB8]
       mov      dword ptr [rbp-0x17C], r9d
       jb       G_M000_IG101
 
G_M000_IG21:                ;; offset=0x04F9
       mov      r10d, dword ptr [rbp+0x18]
       test     r10d, r10d
       je       SHORT G_M000_IG23
 
G_M000_IG22:                ;; offset=0x0502
       test     r14d, r14d
       je       SHORT G_M000_IG23
       mov      r9, r13
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xC0], r9
       mov      r9d, dword ptr [rbp+0x18]
       mov      r10d, r9d
       shl      r10, 2
       cmp      qword ptr [rbp-0xC0], r10
       jb       G_M000_IG101
       mov      r10d, r14d
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xC0]
       mov      dword ptr [rbp+0x18], r9d
       jb       G_M000_IG101
 
G_M000_IG23:                ;; offset=0x0548
       mov      r10d, edx
       test     r10d, r10d
       je       SHORT G_M000_IG25
 
G_M000_IG24:                ;; offset=0x0550
       test     edi, edi
       je       SHORT G_M000_IG25
       mov      r10, r12
       sub      r10, qword ptr [rbp-0x198]
       mov      qword ptr [rbp-0xC8], r10
       mov      r10d, edx
       shl      r10, 2
       cmp      qword ptr [rbp-0xC8], r10
       jb       G_M000_IG101
       mov      r10d, edi
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xC8]
       jb       G_M000_IG101
 
G_M000_IG25:                ;; offset=0x0590
       mov      r10d, esi
       test     r10d, r10d
       je       SHORT G_M000_IG27
 
G_M000_IG26:                ;; offset=0x0598
       test     edi, edi
       je       SHORT G_M000_IG27
       mov      r9, r12
       sub      r9, qword ptr [rbp-0x1A0]
       mov      qword ptr [rbp-0xD0], r9
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0xD0], r9
       jb       G_M000_IG101
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xD0]
       jb       G_M000_IG101
 
G_M000_IG27:                ;; offset=0x05D8
       mov      r10d, dword ptr [rbp-0x17C]
       test     r10d, r10d
       je       SHORT G_M000_IG29
 
G_M000_IG28:                ;; offset=0x05E4
       test     edi, edi
       je       SHORT G_M000_IG29
       mov      r9, r12
       sub      r9, qword ptr [rbp-0x1A8]
       mov      qword ptr [rbp-0xD8], r9
       mov      r9d, dword ptr [rbp-0x17C]
       mov      r10d, r9d
       shl      r10, 2
       cmp      qword ptr [rbp-0xD8], r10
       jb       G_M000_IG101
       mov      r10d, edi
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xD8]
       mov      dword ptr [rbp-0x17C], r9d
       jb       G_M000_IG101
 
G_M000_IG29:                ;; offset=0x0632
       mov      r10d, dword ptr [rbp+0x18]
       test     r10d, r10d
       je       SHORT G_M000_IG31
 
G_M000_IG30:                ;; offset=0x063B
       test     edi, edi
       je       SHORT G_M000_IG31
       mov      r9, r12
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xE0], r9
       mov      r9d, dword ptr [rbp+0x18]
       mov      r10d, r9d
       shl      r10, 2
       cmp      qword ptr [rbp-0xE0], r10
       jb       G_M000_IG101
       mov      r10d, edi
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xE0]
       mov      dword ptr [rbp+0x18], r9d
       jb       G_M000_IG101
 
G_M000_IG31:                ;; offset=0x0680
       mov      r10d, edx
       test     r10d, r10d
       je       SHORT G_M000_IG33
 
G_M000_IG32:                ;; offset=0x0688
       test     r8d, r8d
       je       SHORT G_M000_IG33
       mov      r10, r11
       sub      r10, qword ptr [rbp-0x198]
       mov      qword ptr [rbp-0xE8], r10
       mov      r10d, edx
       shl      r10, 2
       cmp      qword ptr [rbp-0xE8], r10
       jb       G_M000_IG101
       mov      r10d, r8d
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xE8]
       jb       G_M000_IG101
 
G_M000_IG33:                ;; offset=0x06C9
       mov      r10d, esi
       test     r10d, r10d
       je       SHORT G_M000_IG35
 
G_M000_IG34:                ;; offset=0x06D1
       test     r8d, r8d
       je       SHORT G_M000_IG35
       mov      r9, r11
       sub      r9, qword ptr [rbp-0x1A0]
       mov      qword ptr [rbp-0xF0], r9
       mov      dword ptr [rbp-0x178], esi
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0xF0], r9
       jb       G_M000_IG101
       mov      r9d, r8d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xF0]
       mov      esi, dword ptr [rbp-0x178]
       jb       G_M000_IG101
 
G_M000_IG35:                ;; offset=0x071E
       mov      r10d, dword ptr [rbp-0x17C]
       test     r10d, r10d
       je       SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x072A
       test     r8d, r8d
       je       SHORT G_M000_IG37
       mov      r9, r11
       sub      r9, qword ptr [rbp-0x1A8]
       mov      qword ptr [rbp-0xF8], r9
       mov      r9d, dword ptr [rbp-0x17C]
       mov      r10d, r9d
       shl      r10, 2
       cmp      qword ptr [rbp-0xF8], r10
       jb       G_M000_IG101
       mov      r10d, r8d
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0xF8]
       mov      dword ptr [rbp-0x17C], r9d
       jb       G_M000_IG101
 
G_M000_IG37:                ;; offset=0x0779
       mov      r10d, dword ptr [rbp+0x18]
       test     r10d, r10d
       je       SHORT G_M000_IG39
 
G_M000_IG38:                ;; offset=0x0782
       test     r8d, r8d
       je       SHORT G_M000_IG39
       mov      r9, r11
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0x100], r9
       mov      r9d, dword ptr [rbp+0x18]
       mov      r10d, r9d
       shl      r10, 2
       cmp      qword ptr [rbp-0x100], r10
       jb       G_M000_IG101
       mov      r10d, r8d
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0x100]
       mov      dword ptr [rbp+0x18], r9d
       jb       G_M000_IG101
 
G_M000_IG39:                ;; offset=0x07C8
       mov      r10d, r14d
       test     r10d, r10d
       je       SHORT G_M000_IG41
 
G_M000_IG40:                ;; offset=0x07D0
       test     edi, edi
       je       SHORT G_M000_IG41
       mov      r10, r12
       sub      r10, r13
       mov      qword ptr [rbp-0x108], r10
       mov      r10d, r14d
       shl      r10, 2
       cmp      qword ptr [rbp-0x108], r10
       jb       G_M000_IG101
       mov      r10d, edi
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0x108]
       jb       G_M000_IG101
 
G_M000_IG41:                ;; offset=0x080C
       mov      r10d, r14d
       test     r10d, r10d
       je       SHORT G_M000_IG43
 
G_M000_IG42:                ;; offset=0x0814
       test     r8d, r8d
       je       SHORT G_M000_IG43
       mov      r10, r11
       sub      r10, r13
       mov      qword ptr [rbp-0x110], r10
       mov      r10d, r14d
       shl      r10, 2
       cmp      qword ptr [rbp-0x110], r10
       jb       G_M000_IG101
       mov      r10d, r8d
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0x110]
       jb       G_M000_IG101
 
G_M000_IG43:                ;; offset=0x0851
       mov      r10d, r14d
       test     r10d, r10d
       je       SHORT G_M000_IG45
 
G_M000_IG44:                ;; offset=0x0859
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG45
       mov      r10, bword ptr [rbp+0x20]
       sub      r10, r13
       mov      qword ptr [rbp-0x118], r10
       mov      r10d, r14d
       shl      r10, 2
       cmp      qword ptr [rbp-0x118], r10
       jb       G_M000_IG101
       mov      r10d, dword ptr [rbp+0x28]
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0x118]
       jb       G_M000_IG101
 
G_M000_IG45:                ;; offset=0x0899
       mov      r10d, edi
       test     r10d, r10d
       je       SHORT G_M000_IG47
 
G_M000_IG46:                ;; offset=0x08A1
       test     r8d, r8d
       je       SHORT G_M000_IG47
       mov      r10, r11
       sub      r10, r12
       mov      qword ptr [rbp-0x120], r10
       mov      r10d, edi
       shl      r10, 2
       cmp      qword ptr [rbp-0x120], r10
       jb       G_M000_IG101
       mov      r10d, r8d
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0x120]
       jb       G_M000_IG101
 
G_M000_IG47:                ;; offset=0x08DE
       mov      r10d, edi
       test     r10d, r10d
       je       SHORT G_M000_IG49
 
G_M000_IG48:                ;; offset=0x08E6
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG49
       mov      r10, bword ptr [rbp+0x20]
       sub      r10, r12
       mov      qword ptr [rbp-0x128], r10
       mov      dword ptr [rbp+0x48], edi
       mov      r10d, edi
       shl      r10, 2
       cmp      qword ptr [rbp-0x128], r10
       jb       G_M000_IG101
       mov      r10d, dword ptr [rbp+0x28]
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0x128]
       mov      edi, dword ptr [rbp+0x48]
       jb       G_M000_IG101
 
G_M000_IG49:                ;; offset=0x092C
       mov      r10d, r8d
       test     r10d, r10d
       je       SHORT G_M000_IG51
 
G_M000_IG50:                ;; offset=0x0934
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG51
       mov      r10, bword ptr [rbp+0x20]
       mov      bword ptr [rbp+0x50], r11
       sub      r10, r11
       mov      qword ptr [rbp-0x130], r10
       mov      dword ptr [rbp+0x58], r8d
       mov      r10d, r8d
       shl      r10, 2
       cmp      qword ptr [rbp-0x130], r10
       jb       G_M000_IG101
       mov      r10d, dword ptr [rbp+0x28]
       shl      r10, 2
       neg      r10
       cmp      r10, qword ptr [rbp-0x130]
       mov      r8d, dword ptr [rbp+0x58]
       mov      r11, bword ptr [rbp+0x50]
       jb       G_M000_IG101
 
G_M000_IG51:                ;; offset=0x0984
       cmp      dword ptr [rbp+0x80], 16
       je       G_M000_IG103
       mov      dword ptr [rbp-0x180], edx
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r10, r10
       cmp      dword ptr [rbp-0x180], 0
       cmovne   r10, bword ptr [rbp-0x198]
       mov      bword ptr [rbp-0x138], r10
       mov      qword ptr [rbp-0x140], r10
       xor      r9d, r9d
       cmp      edx, 8
       mov      dword ptr [rbp-0x178], esi
       jl       G_M000_IG102
       align    [0 bytes for IG52]
 
G_M000_IG52:                ;; offset=0x09D2
       mov      r10d, r9d
       sar      r10d, 31
       and      r10d, 7
       add      r10d, r9d
       sar      r10d, 3
       movsxd   r10, r10d
       shl      r10, 5
       mov      rsi, qword ptr [rbp-0x140]
       vpand    ymm1, ymm0, ymmword ptr [r10+rsi]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG104
       add      r9d, 8
       lea      r10d, [r9+0x08]
       cmp      r10d, edx
       jg       SHORT G_M000_IG57
 
G_M000_IG53:                ;; offset=0x0A14
       jmp      SHORT G_M000_IG52
 
G_M000_IG54:                ;; offset=0x0A16
       mov      dword ptr [rbp-0x17C], r9d
       cmp      r9d, ebx
       mov      r9d, dword ptr [rbp-0x17C]
       jne      G_M000_IG99
       jmp      G_M000_IG06
 
G_M000_IG55:                ;; offset=0x0A32
       cmp      r10d, dword ptr [rbp-0x40]
       jne      G_M000_IG99
       jmp      G_M000_IG07
 
G_M000_IG56:                ;; offset=0x0A41
       mov      dword ptr [rbp+0x18], r10d
       jmp      G_M000_IG17
       align    [0 bytes for IG59]
 
G_M000_IG57:                ;; offset=0x0A4A
       mov      dword ptr [rbp-0x174], edx
       cmp      r9d, edx
       jl       G_M000_IG105
       xor      esi, esi
       mov      bword ptr [rbp-0x138], rsi
       mov      esi, 1
 
G_M000_IG58:                ;; offset=0x0A67
       xor      r9, r9
       mov      bword ptr [rbp-0x138], r9
       test     esi, esi
       je       G_M000_IG119
       mov      r10d, dword ptr [rbp-0x178]
       mov      esi, r10d
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r9, r9
       test     esi, esi
       cmovne   r9, bword ptr [rbp-0x1A0]
       mov      bword ptr [rbp-0x148], r9
       mov      qword ptr [rbp-0x150], r9
       xor      esi, esi
       mov      dword ptr [rbp-0x178], r10d
       cmp      r10d, 8
       jl       G_M000_IG107
       jmp      SHORT G_M000_IG60
 
G_M000_IG59:                ;; offset=0x0ABC
       mov      dword ptr [rbp-0x178], r10d
 
G_M000_IG60:                ;; offset=0x0AC3
       mov      r9d, esi
       sar      r9d, 31
       and      r9d, 7
       add      r9d, esi
       sar      r9d, 3
       movsxd   r9, r9d
       shl      r9, 5
       mov      r10, qword ptr [rbp-0x150]
       vpand    ymm1, ymm0, ymmword ptr [r9+r10]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG108
       add      esi, 8
       lea      r9d, [rsi+0x08]
       mov      r10d, dword ptr [rbp-0x178]
       cmp      r9d, r10d
       jle      SHORT G_M000_IG59
 
G_M000_IG61:                ;; offset=0x0B0B
       mov      dword ptr [rbp-0x178], r10d
       cmp      esi, r10d
       jl       G_M000_IG109
       xor      esi, esi
       mov      bword ptr [rbp-0x148], rsi
       mov      esi, 1
 
G_M000_IG62:                ;; offset=0x0B29
       xor      r9, r9
       mov      bword ptr [rbp-0x148], r9
       test     esi, esi
       je       G_M000_IG119
       mov      r9d, dword ptr [rbp-0x17C]
       mov      dword ptr [rbp-0x184], r9d
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      rsi, rsi
       cmp      dword ptr [rbp-0x184], 0
       cmovne   rsi, bword ptr [rbp-0x1A8]
       mov      bword ptr [rbp-0x158], rsi
       mov      qword ptr [rbp-0x160], rsi
       xor      r9d, r9d
       cmp      dword ptr [rbp-0x17C], 8
       jl       G_M000_IG110
       jmp      SHORT G_M000_IG64
       align    [0 bytes for IG63]
 
G_M000_IG63:                ;; offset=0x0B83
       mov      dword ptr [rbp-0x17C], r10d
 
G_M000_IG64:                ;; offset=0x0B8A
       mov      esi, r9d
       sar      esi, 31
       and      esi, 7
       add      esi, r9d
       sar      esi, 3
       movsxd   rsi, esi
       shl      rsi, 5
       mov      r10, qword ptr [rbp-0x160]
       vpand    ymm1, ymm0, ymmword ptr [rsi+r10]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG111
       add      r9d, 8
       lea      esi, [r9+0x08]
       mov      r10d, dword ptr [rbp-0x17C]
       cmp      esi, r10d
       jle      SHORT G_M000_IG63
 
G_M000_IG65:                ;; offset=0x0BD0
       mov      dword ptr [rbp-0x17C], r10d
       cmp      r9d, r10d
       jl       G_M000_IG112
       xor      esi, esi
       mov      bword ptr [rbp-0x158], rsi
       mov      esi, 1
 
G_M000_IG66:                ;; offset=0x0BEE
       xor      r9, r9
       mov      bword ptr [rbp-0x158], r9
       test     esi, esi
       je       G_M000_IG119
       mov      r9d, dword ptr [rbp+0x18]
       mov      dword ptr [rbp-0x188], r9d
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      rsi, rsi
       cmp      dword ptr [rbp-0x188], 0
       cmovne   rsi, bword ptr [rbp+0x10]
       mov      bword ptr [rbp-0x168], rsi
       xor      r9d, r9d
       cmp      dword ptr [rbp+0x18], 8
       jl       SHORT G_M000_IG68
       align    [0 bytes for IG67]
 
G_M000_IG67:                ;; offset=0x0C32
       mov      r10d, r9d
       sar      r10d, 31
       and      r10d, 7
       add      r10d, r9d
       sar      r10d, 3
       movsxd   r10, r10d
       shl      r10, 5
       vpand    ymm1, ymm0, ymmword ptr [r10+rsi]
       vpcmpeqd ymm1, ymm1, ymm0
       vptest   ymm1, ymm1
       jne      G_M000_IG113
       add      r9d, 8
       lea      r10d, [r9+0x08]
       cmp      r10d, dword ptr [rbp+0x18]
       jle      SHORT G_M000_IG67
 
G_M000_IG68:                ;; offset=0x0C6E
       cmp      r9d, dword ptr [rbp+0x18]
       jl       G_M000_IG114
       xor      esi, esi
       mov      bword ptr [rbp-0x168], rsi
       mov      esi, 1
 
G_M000_IG69:                ;; offset=0x0C86
       xor      r9, r9
       mov      bword ptr [rbp-0x168], r9
       test     esi, esi
       je       G_M000_IG119
       lea      r9d, [rcx+0x01]
       mov      esi, r9d
       shr      esi, 31
       add      r9d, esi
       sar      r9d, 1
       mov      dword ptr [rbp-0x44], r9d
       mov      esi, dword ptr [rbp-0x190]
       shr      esi, 31
       add      esi, dword ptr [rbp-0x190]
       sar      esi, 1
       imul     esi, r9d
       jo       G_M000_IG121
       mov      dword ptr [rbp-0x48], esi
       xor      r10d, r10d
       cmp      r10d, esi
       jl       G_M000_IG75
       jmp      G_M000_IG115
 
G_M000_IG70:                ;; offset=0x0CDB
       mov      rdi, r12
       mov      dword ptr [rbp+0x48], r8d
       mov      esi, r8d
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG119
       mov      eax, dword ptr [rbp-0x4C]
       mov      dword ptr [rsp], eax
       mov      edi, dword ptr [rbp-0x50]
       mov      dword ptr [rsp+0x08], edi
       mov      rdi, qword ptr [rbp-0x58]
       mov      rsi, qword ptr [rbp-0x60]
       mov      edx, ebx
       mov      ecx, dword ptr [rbp+0x70]
       mov      r8d, dword ptr [rbp+0x78]
       mov      r9d, dword ptr [rbp-0x44]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
       xor      edi, edi
       mov      bword ptr [rbp-0x68], rdi
 
G_M000_IG71:                ;; offset=0x0D21
       mov      bword ptr [rbp-0x70], rdi
 
G_M000_IG72:                ;; offset=0x0D25
       mov      bword ptr [rbp-0x78], rdi
 
G_M000_IG73:                ;; offset=0x0D29
       mov      bword ptr [rbp-0x80], rdi
       mov      eax, dword ptr [rbp-0x4C]
       add      eax, 8
       mov      edi, dword ptr [rbp-0x48]
       cmp      eax, edi
       mov      esi, edi
       mov      r10d, eax
       mov      r9d, dword ptr [rbp-0x44]
       jge      G_M000_IG94
 
G_M000_IG74:                ;; offset=0x0D47
       mov      eax, dword ptr [rbp+0x70]
       mov      ecx, dword ptr [rbp+0x78]
       mov      edi, dword ptr [rbp+0x48]
       mov      r8d, dword ptr [rbp+0x58]
       mov      r11, bword ptr [rbp+0x50]
 
G_M000_IG75:                ;; offset=0x0D58
       mov      edx, esi
       sub      edx, r10d
       cmp      edx, 8
       jl       G_M000_IG116
       mov      edx, 8
       mov      dword ptr [rbp+0x48], edi
       mov      dword ptr [rbp+0x58], r8d
       mov      bword ptr [rbp+0x50], r11
 
G_M000_IG76:                ;; offset=0x0D76
       mov      dword ptr [rbp-0x50], edx
       mov      dword ptr [rbp+0x78], ecx
       mov      dword ptr [rsp], ecx
       mov      dword ptr [rsp+0x08], r9d
       mov      dword ptr [rbp-0x4C], r10d
       mov      dword ptr [rsp+0x10], r10d
       mov      dword ptr [rsp+0x18], edx
       mov      rdi, bword ptr [rbp-0x198]
       mov      esi, dword ptr [rbp-0x174]
       mov      rdx, r13
       mov      ecx, r14d
       mov      r8d, r15d
       mov      dword ptr [rbp+0x70], eax
       mov      r9d, eax
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      edi, r14d
       xor      rsi, rsi
       mov      bword ptr [rbp-0x170], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG77:                ;; offset=0x0DC8
       test     edi, edi
       je       SHORT G_M000_IG78
       mov      rsi, r13
 
G_M000_IG78:                ;; offset=0x0DCF
       mov      bword ptr [rbp-0x170], rsi
       xor      edi, edi
       cmp      r14d, 8
       jl       SHORT G_M000_IG80
       align    [2 bytes for IG79]
 
G_M000_IG79:                ;; offset=0x0DE0
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
       jne      G_M000_IG117
       add      edi, 8
       lea      eax, [rdi+0x08]
       cmp      eax, r14d
       jle      SHORT G_M000_IG79
       align    [0 bytes for IG80]
 
G_M000_IG80:                ;; offset=0x0E12
       cmp      edi, r14d
       jl       G_M000_IG118
       xor      edi, edi
       mov      bword ptr [rbp-0x170], rdi
       mov      edi, 1
 
G_M000_IG81:                ;; offset=0x0E29
       xor      rsi, rsi
       mov      bword ptr [rbp-0x170], rsi
       test     edi, edi
       je       G_M000_IG119
       xor      rdi, rdi
       test     r14d, r14d
       je       SHORT G_M000_IG82
       mov      rdi, r13
 
G_M000_IG82:                ;; offset=0x0E44
       mov      bword ptr [rbp-0x68], rdi
       xor      rsi, rsi
       cmp      dword ptr [rbp-0x178], 0
       je       SHORT G_M000_IG83
       mov      rcx, bword ptr [rbp-0x1A0]
       mov      rsi, rcx
 
G_M000_IG83:                ;; offset=0x0E5D
       mov      bword ptr [rbp-0x70], rsi
       xor      rdx, rdx
       mov      r8d, dword ptr [rbp+0x48]
       test     r8d, r8d
       je       SHORT G_M000_IG84
       mov      bword ptr [rbp+0x40], r12
       mov      rdx, r12
       mov      r12, bword ptr [rbp+0x40]
 
G_M000_IG84:                ;; offset=0x0E77
       mov      bword ptr [rbp-0x78], rdx
       mov      qword ptr [rbp-0x58], rdx
       xor      r9, r9
       cmp      dword ptr [rbp+0x58], 0
       je       SHORT G_M000_IG85
       mov      r11, bword ptr [rbp+0x50]
       mov      r9, r11
 
G_M000_IG85:                ;; offset=0x0E8F
       mov      bword ptr [rbp-0x80], r9
       mov      qword ptr [rbp-0x60], r9
       xor      ecx, ecx
       movsxd   rax, ebx
       shl      rax, 2
       jmp      SHORT G_M000_IG88
       align    [0 bytes for IG91]
 
G_M000_IG86:                ;; offset=0x0EA2
       inc      ecx
       cmp      ecx, 16
       mov      r12, bword ptr [rbp+0x40]
       jge      G_M000_IG70
 
G_M000_IG87:                ;; offset=0x0EB1
       mov      rdx, qword ptr [rbp-0x58]
 
G_M000_IG88:                ;; offset=0x0EB5
       xor      r11d, r11d
       cmp      r11d, ebx
       mov      bword ptr [rbp+0x40], r12
       jge      SHORT G_M000_IG86
 
G_M000_IG89:                ;; offset=0x0EC1
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r10d, ecx
       imul     r10d, r15d
       mov      r9d, r10d
       imul     r9d, ebx
       movsxd   r9, r9d
       shl      r9, 2
       add      r9, rsi
       movsxd   r12, r11d
       lea      r9, [r9+4*r12]
       shl      r10d, 3
       movsxd   r10, r10d
       lea      r10, [rdi+4*r10]
       test     r15d, r15d
       jle      SHORT G_M000_IG92
 
G_M000_IG90:                ;; offset=0x0F10
       mov      r12d, r15d
 
G_M000_IG91:                ;; offset=0x0F13
       vmovups  ymm8, ymmword ptr [r9]
       vbroadcastss ymm9, dword ptr [r10]
       vfmadd231ps ymm0, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r10+0x04]
       vfmadd231ps ymm1, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r10+0x08]
       vfmadd231ps ymm2, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r10+0x0C]
       vfmadd231ps ymm3, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r10+0x10]
       vfmadd231ps ymm4, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r10+0x14]
       vfmadd231ps ymm5, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r10+0x18]
       vfmadd231ps ymm6, ymm8, ymm9
       vbroadcastss ymm9, dword ptr [r10+0x1C]
       vfmadd231ps ymm7, ymm8, ymm9
       add      r9, rax
       add      r10, 32
       dec      r12d
       jne      SHORT G_M000_IG91
 
G_M000_IG92:                ;; offset=0x0F7B
       mov      r9d, ecx
       imul     r9d, ebx
       add      r9d, r11d
       shl      r9d, 3
       movsxd   r9, r9d
       lea      r9, [rdx+4*r9]
       vmovups  ymmword ptr [r9], ymm0
       vmovups  ymmword ptr [r9+0x20], ymm1
       vmovups  ymmword ptr [r9+0x40], ymm2
       vmovups  ymmword ptr [r9+0x60], ymm3
       vmovups  ymmword ptr [r9+0x80], ymm4
       vmovups  ymmword ptr [r9+0xA0], ymm5
       vmovups  ymmword ptr [r9+0xC0], ymm6
       vmovups  ymmword ptr [r9+0xE0], ymm7
       add      r11d, 8
       cmp      r11d, ebx
       jge      G_M000_IG86
 
G_M000_IG93:                ;; offset=0x0FD8
       mov      rdx, qword ptr [rbp-0x58]
       jmp      G_M000_IG89
 
G_M000_IG94:                ;; offset=0x0FE1
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
       mov      rdi, bword ptr [rbp-0x1A8]
       mov      esi, dword ptr [rbp-0x17C]
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
       mov      r8, bword ptr [rbp-0x1A8]
       mov      r9d, dword ptr [rbp-0x17C]
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:UnpackEpilogue(System.ReadOnlySpan`1[float],System.Span`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],int,int,int,bool)]
       mov      eax, 1
 
G_M000_IG95:                ;; offset=0x1097
       vzeroupper 
       add      rsp, 440
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG96:                ;; offset=0x10AC
       cmp      dword ptr [rbp+0x80], 16
       je       G_M000_IG03
 
G_M000_IG97:                ;; offset=0x10B9
       mov      r8d, 1
       jmp      G_M000_IG04
 
G_M000_IG98:                ;; offset=0x10C4
       mov      rdi, 0x7C0BEAA9CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0xC72
       mov      rsi, 0x7C0BEAB51A30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG99:                ;; offset=0x1100
       mov      rdi, 0x7C0BEAA9CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0xC0A
       mov      rsi, 0x7C0BEAB51A30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG100:                ;; offset=0x113C
       call     [System.ThrowHelper:ThrowArgumentOutOfRangeException()]
       int3     
 
G_M000_IG101:                ;; offset=0x1143
       mov      rdi, 0x7C0BEAA9CD28
       call     CORINFO_HELP_NEWSFAST
       mov      r13, rax
       mov      edi, 0xC38
       mov      rsi, 0x7C0BEAB51A30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, r13
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, r13
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG102:                ;; offset=0x117F
       mov      rsi, qword ptr [rbp-0x140]
       jmp      G_M000_IG57
 
G_M000_IG103:                ;; offset=0x118B
       mov      rdi, 0x7C0BEB39DBD8
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      rdi, rbx
       call     [System.PlatformNotSupportedException:.ctor():this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG104:                ;; offset=0x11AF
       xor      esi, esi
       mov      dword ptr [rbp-0x174], edx
       jmp      G_M000_IG58
 
G_M000_IG105:                ;; offset=0x11BC
       movsxd   r10, r9d
       mov      r10d, dword ptr [rsi+4*r10]
       mov      dword ptr [rbp-0x1AC], r10d
       mov      r10d, 0x7F800000
       mov      dword ptr [rbp-0x1B0], r10d
       mov      r10d, dword ptr [rbp-0x1AC]
       andn     r10d, r10d, dword ptr [rbp-0x1B0]
       je       SHORT G_M000_IG106
       inc      r9d
       mov      edx, dword ptr [rbp-0x174]
       jmp      G_M000_IG57
 
G_M000_IG106:                ;; offset=0x11F7
       mov      edx, dword ptr [rbp-0x174]
       jmp      SHORT G_M000_IG104
 
G_M000_IG107:                ;; offset=0x11FF
       mov      r10d, dword ptr [rbp-0x178]
       jmp      G_M000_IG61
 
G_M000_IG108:                ;; offset=0x120B
       xor      esi, esi
       jmp      G_M000_IG62
 
G_M000_IG109:                ;; offset=0x1212
       movsxd   r9, esi
       mov      r10, qword ptr [rbp-0x150]
       mov      r9d, dword ptr [r10+4*r9]
       mov      dword ptr [rbp-0x1B0], r9d
       mov      r9d, 0x7F800000
       mov      dword ptr [rbp-0x1AC], r9d
       mov      r9d, dword ptr [rbp-0x1B0]
       andn     r9d, r9d, dword ptr [rbp-0x1AC]
       je       SHORT G_M000_IG108
       inc      esi
       mov      r10d, dword ptr [rbp-0x178]
       jmp      G_M000_IG61
 
G_M000_IG110:                ;; offset=0x1254
       mov      r10d, dword ptr [rbp-0x17C]
       jmp      G_M000_IG65
 
G_M000_IG111:                ;; offset=0x1260
       xor      esi, esi
       jmp      G_M000_IG66
 
G_M000_IG112:                ;; offset=0x1267
       movsxd   rsi, r9d
       mov      r10, qword ptr [rbp-0x160]
       mov      esi, dword ptr [r10+4*rsi]
       mov      dword ptr [rbp-0x1AC], esi
       mov      esi, 0x7F800000
       mov      dword ptr [rbp-0x1B0], esi
       mov      esi, dword ptr [rbp-0x1AC]
       andn     esi, esi, dword ptr [rbp-0x1B0]
       je       SHORT G_M000_IG111
       inc      r9d
       mov      r10d, dword ptr [rbp-0x17C]
       jmp      G_M000_IG65
 
G_M000_IG113:                ;; offset=0x12A6
       xor      esi, esi
       jmp      G_M000_IG69
 
G_M000_IG114:                ;; offset=0x12AD
       movsxd   r10, r9d
       mov      r10d, dword ptr [rsi+4*r10]
       mov      dword ptr [rbp-0x1B0], r10d
       mov      r10d, 0x7F800000
       mov      dword ptr [rbp-0x1AC], r10d
       mov      r10d, dword ptr [rbp-0x1B0]
       andn     r10d, r10d, dword ptr [rbp-0x1AC]
       je       SHORT G_M000_IG113
       inc      r9d
       jmp      G_M000_IG68
 
G_M000_IG115:                ;; offset=0x12E2
       mov      dword ptr [rbp+0x78], ecx
       mov      dword ptr [rbp+0x70], eax
       mov      dword ptr [rbp+0x58], r8d
       mov      bword ptr [rbp+0x50], r11
       jmp      G_M000_IG94
 
G_M000_IG116:                ;; offset=0x12F5
       mov      dword ptr [rbp+0x48], edi
       mov      dword ptr [rbp+0x58], r8d
       mov      bword ptr [rbp+0x50], r11
       jmp      G_M000_IG76
 
G_M000_IG117:                ;; offset=0x1305
       xor      edi, edi
       jmp      G_M000_IG81
 
G_M000_IG118:                ;; offset=0x130C
       movsxd   rax, edi
       mov      eax, dword ptr [rsi+4*rax]
       mov      ecx, 0x7F800000
       andn     eax, eax, ecx
       je       SHORT G_M000_IG117
       inc      edi
       jmp      G_M000_IG80
 
G_M000_IG119:                ;; offset=0x1325
       xor      eax, eax
 
G_M000_IG120:                ;; offset=0x1327
       vzeroupper 
       add      rsp, 440
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG121:                ;; offset=0x133C
       call     CORINFO_HELP_OVERFLOW
       int3     
 
RWD00  	dd	7F800000h		;       inf

; Total bytes of code 4930

