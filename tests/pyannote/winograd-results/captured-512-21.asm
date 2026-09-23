; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:ExecuteWinograd(System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.ReadOnlySpan`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],System.Span`1[float],int,int,int,int,int,bool):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 100
; 45 inlinees with PGO data; 165 single block inlinees; 0 inlinees without PGO data

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
       mov      r15d, dword ptr [rbp+0x60]
       mov      ebx, dword ptr [rbp+0x68]
       mov      r10d, dword ptr [rbp+0x70]
       mov      r12d, dword ptr [rbp+0x78]
       mov      r14d, dword ptr [rbp+0x38]
       mov      r13d, dword ptr [rbp+0x48]
 
G_M000_IG02:                ;; offset=0x0095
       mov      r8d, dword ptr [rbp+0x80]
       cmp      r8d, 8
       je       SHORT G_M000_IG03
       cmp      r8d, 16
       jne      G_M000_IG112
 
G_M000_IG03:                ;; offset=0x00AC
       cmp      r15d, 16
       jl       G_M000_IG112
       test     r15b, 15
       jne      G_M000_IG112
       cmp      ebx, 32
       jl       G_M000_IG112
       test     bl, 15
       jne      G_M000_IG112
       test     r10d, r10d
       jle      G_M000_IG112
       test     r12d, r12d
       setle    r9b
       movzx    r9, r9b
 
G_M000_IG04:                ;; offset=0x00E6
       test     r9d, r9d
       jne      G_M000_IG113
       mov      r9d, r10d
       add      r9d, 2
       jo       G_M000_IG140
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, r12d
       add      r9d, 2
       jo       G_M000_IG140
       imul     r9d, dword ptr [rbp-0x1BC]
       jo       G_M000_IG140
       imul     r9d, r15d
       jo       G_M000_IG140
       mov      r9d, r10d
       add      r9d, 1
       jo       G_M000_IG140
       mov      dword ptr [rbp-0x198], r9d
       sub      r9d, 1
       jo       G_M000_IG140
       imul     r9d, ebx
       jo       G_M000_IG140
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, r12d
       add      r9d, 1
       jo       G_M000_IG140
       sub      r9d, 1
       jo       G_M000_IG140
       imul     r9d, dword ptr [rbp-0x1BC]
       jo       G_M000_IG140
       mov      dword ptr [rbp+0x80], r8d
       imul     r9d, r8d, 2
       jo       G_M000_IG140
       mov      dword ptr [rbp-0x190], r9d
       add      r9d, ebx
       jo       G_M000_IG140
       mov      eax, r9d
       sub      eax, 1
       jo       G_M000_IG140
       mov      r9d, dword ptr [rbp-0x190]
       cdq      
       idiv     edx:eax, r9d
       imul     r9d, eax
       jo       G_M000_IG140
       imul     r9d, r15d
 
G_M000_IG05:                ;; offset=0x01C2
       jo       G_M000_IG140
       imul     r9d, r9d, 9
       jo       G_M000_IG140
       lea      r9, [rbp-0x40]
       mov      qword ptr [rsp], r9
       lea      r8, [rbp-0x30]
       lea      r9, [rbp-0x38]
       mov      edi, r15d
       mov      esi, ebx
       mov      dword ptr [rbp+0x70], r10d
       mov      edx, r10d
       mov      ecx, r12d
       call     [Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool]
       test     eax, eax
       je       G_M000_IG138
       mov      eax, dword ptr [rbp+0x70]
       mov      edi, r15d
       imul     edi, eax
       jo       G_M000_IG140
       imul     edi, r12d
       jo       G_M000_IG140
       mov      ecx, dword ptr [rbp-0x17C]
       cmp      edi, ecx
       jne      G_M000_IG114
       imul     edi, r15d, 16
       jo       G_M000_IG140
       imul     edi, ebx
       jo       G_M000_IG140
       mov      edx, dword ptr [rbp-0x180]
       cmp      edi, edx
       jne      G_M000_IG114
       mov      esi, dword ptr [rbp+0x28]
       cmp      esi, dword ptr [rbp-0x40]
       jne      G_M000_IG114
       mov      r8d, dword ptr [rbp-0x184]
       test     r8d, r8d
       je       SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x025F
       mov      dword ptr [rbp-0x184], r8d
       cmp      r8d, ebx
       mov      r8d, dword ptr [rbp-0x184]
       jne      G_M000_IG114
 
G_M000_IG07:                ;; offset=0x0276
       mov      r9d, dword ptr [rbp+0x18]
       test     r9d, r9d
       je       SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x027F
       cmp      r9d, dword ptr [rbp-0x40]
       jne      G_M000_IG114
 
G_M000_IG09:                ;; offset=0x0289
       cmp      r14d, dword ptr [rbp-0x30]
       jl       G_M000_IG114
       cmp      r13d, dword ptr [rbp-0x38]
       jl       G_M000_IG114
       mov      r10d, dword ptr [rbp+0x58]
       cmp      r10d, dword ptr [rbp-0x40]
       jl       G_M000_IG114
       mov      edi, dword ptr [rbp-0x30]
       cmp      edi, r14d
       ja       G_M000_IG115
       mov      r14, bword ptr [rbp+0x30]
       mov      r11d, edi
       mov      edi, dword ptr [rbp-0x38]
       cmp      edi, r13d
       ja       G_M000_IG115
       mov      r13, bword ptr [rbp+0x40]
       mov      esi, dword ptr [rbp-0x40]
       cmp      esi, r10d
       ja       G_M000_IG115
       mov      r10, bword ptr [rbp+0x50]
       test     ecx, ecx
       je       SHORT G_M000_IG11
 
G_M000_IG10:                ;; offset=0x02E2
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG11
       mov      dword ptr [rbp-0x184], r8d
       mov      r8, bword ptr [rbp+0x20]
       sub      r8, qword ptr [rbp-0x1A8]
       mov      qword ptr [rbp-0x88], r8
       mov      r8d, ecx
       shl      r8, 2
       cmp      qword ptr [rbp-0x88], r8
       jb       G_M000_IG116
       mov      r8d, dword ptr [rbp+0x28]
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0x88]
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG116
 
G_M000_IG11:                ;; offset=0x0334
       test     edx, edx
       je       SHORT G_M000_IG13
 
G_M000_IG12:                ;; offset=0x0338
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG13
       mov      dword ptr [rbp-0x184], r8d
       mov      r8, bword ptr [rbp+0x20]
       sub      r8, qword ptr [rbp-0x1B0]
       mov      qword ptr [rbp-0x90], r8
       mov      r8d, edx
       shl      r8, 2
       cmp      qword ptr [rbp-0x90], r8
       jb       G_M000_IG116
       mov      r8d, dword ptr [rbp+0x28]
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0x90]
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG116
 
G_M000_IG13:                ;; offset=0x038A
       mov      dword ptr [rbp-0x184], r8d
       test     r8d, r8d
       je       SHORT G_M000_IG15
 
G_M000_IG14:                ;; offset=0x0396
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG15
       mov      r8, bword ptr [rbp+0x20]
       sub      r8, qword ptr [rbp-0x1B8]
       mov      qword ptr [rbp-0x98], r8
       mov      r8d, dword ptr [rbp-0x184]
       shl      r8, 2
       cmp      qword ptr [rbp-0x98], r8
       jb       G_M000_IG116
       mov      r8d, dword ptr [rbp+0x28]
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0x98]
       jb       G_M000_IG116
 
G_M000_IG15:                ;; offset=0x03DE
       test     r9d, r9d
       je       SHORT G_M000_IG17
 
G_M000_IG16:                ;; offset=0x03E3
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG17
       mov      r8, bword ptr [rbp+0x20]
       sub      r8, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xA0], r8
       mov      dword ptr [rbp+0x18], r9d
       mov      r8d, r9d
       shl      r8, 2
       cmp      qword ptr [rbp-0xA0], r8
       jb       G_M000_IG116
       mov      r8d, dword ptr [rbp+0x28]
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xA0]
       mov      r9d, dword ptr [rbp+0x18]
       jb       G_M000_IG116
 
G_M000_IG17:                ;; offset=0x042C
       test     ecx, ecx
       je       SHORT G_M000_IG19
 
G_M000_IG18:                ;; offset=0x0430
       test     r11d, r11d
       je       SHORT G_M000_IG19
       mov      dword ptr [rbp+0x18], r9d
       mov      r8, bword ptr [rbp-0x1A8]
       mov      r9, r14
       sub      r9, r8
       mov      qword ptr [rbp-0xA8], r9
       mov      r9d, ecx
       shl      r9, 2
       cmp      qword ptr [rbp-0xA8], r9
       jb       G_M000_IG116
       mov      r9d, r11d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xA8]
       mov      bword ptr [rbp-0x1A8], r8
       mov      r9d, dword ptr [rbp+0x18]
       jb       G_M000_IG116
 
G_M000_IG19:                ;; offset=0x0483
       test     edx, edx
       je       SHORT G_M000_IG21
 
G_M000_IG20:                ;; offset=0x0487
       test     r11d, r11d
       je       SHORT G_M000_IG21
       mov      dword ptr [rbp+0x18], r9d
       mov      r9, bword ptr [rbp-0x1B0]
       mov      r8, r14
       sub      r8, r9
       mov      qword ptr [rbp-0xB0], r8
       mov      r8d, edx
       shl      r8, 2
       cmp      qword ptr [rbp-0xB0], r8
       jb       G_M000_IG116
       mov      r8d, r11d
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xB0]
       mov      bword ptr [rbp-0x1B0], r9
       jb       G_M000_IG116
       mov      r9d, dword ptr [rbp+0x18]
 
G_M000_IG21:                ;; offset=0x04DA
       mov      r8d, dword ptr [rbp-0x184]
       test     r8d, r8d
       je       SHORT G_M000_IG23
 
G_M000_IG22:                ;; offset=0x04E6
       test     r11d, r11d
       je       SHORT G_M000_IG23
       mov      dword ptr [rbp+0x18], r9d
       mov      r9, r14
       sub      r9, qword ptr [rbp-0x1B8]
       mov      qword ptr [rbp-0xB8], r9
       mov      dword ptr [rbp-0x184], r8d
       mov      r9d, r8d
       shl      r9, 2
       cmp      qword ptr [rbp-0xB8], r9
       jb       G_M000_IG116
       mov      r9d, r11d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xB8]
       mov      r8d, dword ptr [rbp-0x184]
       mov      r9d, dword ptr [rbp+0x18]
       jb       G_M000_IG116
 
G_M000_IG23:                ;; offset=0x053D
       mov      dword ptr [rbp+0x18], r9d
       test     r9d, r9d
       je       SHORT G_M000_IG25
 
G_M000_IG24:                ;; offset=0x0546
       test     r11d, r11d
       je       SHORT G_M000_IG25
       mov      r9, r14
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xC0], r9
       mov      r9d, dword ptr [rbp+0x18]
       shl      r9, 2
       cmp      qword ptr [rbp-0xC0], r9
       jb       G_M000_IG116
       mov      r9d, r11d
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xC0]
       jb       G_M000_IG116
 
G_M000_IG25:                ;; offset=0x0585
       test     ecx, ecx
       je       SHORT G_M000_IG27
 
G_M000_IG26:                ;; offset=0x0589
       test     edi, edi
       je       SHORT G_M000_IG27
       mov      dword ptr [rbp-0x184], r8d
       mov      r9, bword ptr [rbp-0x1A8]
       mov      r8, r13
       sub      r8, r9
       mov      qword ptr [rbp-0xC8], r8
       mov      r8d, ecx
       shl      r8, 2
       cmp      qword ptr [rbp-0xC8], r8
       jb       G_M000_IG116
       mov      r8d, edi
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xC8]
       mov      bword ptr [rbp-0x1A8], r9
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG116
 
G_M000_IG27:                ;; offset=0x05E1
       test     edx, edx
       je       SHORT G_M000_IG29
 
G_M000_IG28:                ;; offset=0x05E5
       test     edi, edi
       je       SHORT G_M000_IG29
       mov      dword ptr [rbp-0x184], r8d
       mov      r9, bword ptr [rbp-0x1B0]
       mov      r8, r13
       sub      r8, r9
       mov      qword ptr [rbp-0xD0], r8
       mov      r8d, edx
       shl      r8, 2
       cmp      qword ptr [rbp-0xD0], r8
       jb       G_M000_IG116
       mov      r8d, edi
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xD0]
       mov      bword ptr [rbp-0x1B0], r9
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG116
 
G_M000_IG29:                ;; offset=0x063D
       test     r8d, r8d
       je       SHORT G_M000_IG31
 
G_M000_IG30:                ;; offset=0x0642
       test     edi, edi
       je       SHORT G_M000_IG31
       mov      r9, r13
       sub      r9, qword ptr [rbp-0x1B8]
       mov      qword ptr [rbp-0xD8], r9
       mov      dword ptr [rbp-0x184], r8d
       mov      r9d, r8d
       shl      r9, 2
       cmp      qword ptr [rbp-0xD8], r9
       jb       G_M000_IG116
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xD8]
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG116
 
G_M000_IG31:                ;; offset=0x0690
       cmp      dword ptr [rbp+0x18], 0
       je       SHORT G_M000_IG33
 
G_M000_IG32:                ;; offset=0x0696
       test     edi, edi
       je       SHORT G_M000_IG33
       mov      r9, r13
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0xE0], r9
       mov      r9d, dword ptr [rbp+0x18]
       shl      r9, 2
       cmp      qword ptr [rbp-0xE0], r9
       jb       G_M000_IG116
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xE0]
       jb       G_M000_IG116
 
G_M000_IG33:                ;; offset=0x06D4
       test     ecx, ecx
       je       SHORT G_M000_IG35
 
G_M000_IG34:                ;; offset=0x06D8
       test     esi, esi
       je       SHORT G_M000_IG35
       mov      dword ptr [rbp-0x184], r8d
       mov      r9, bword ptr [rbp-0x1A8]
       mov      r8, r10
       sub      r8, r9
       mov      qword ptr [rbp-0xE8], r8
       mov      r8d, ecx
       shl      r8, 2
       cmp      qword ptr [rbp-0xE8], r8
       jb       G_M000_IG116
       mov      r8d, esi
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xE8]
       mov      bword ptr [rbp-0x1A8], r9
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG116
 
G_M000_IG35:                ;; offset=0x0730
       test     edx, edx
       je       SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x0734
       test     esi, esi
       je       SHORT G_M000_IG37
       mov      dword ptr [rbp-0x184], r8d
       mov      r9, bword ptr [rbp-0x1B0]
       mov      r8, r10
       sub      r8, r9
       mov      qword ptr [rbp-0xF0], r8
       mov      r8d, edx
       shl      r8, 2
       cmp      qword ptr [rbp-0xF0], r8
       jb       G_M000_IG116
       mov      r8d, esi
       shl      r8, 2
       neg      r8
       cmp      r8, qword ptr [rbp-0xF0]
       mov      bword ptr [rbp-0x1B0], r9
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG116
 
G_M000_IG37:                ;; offset=0x078C
       test     r8d, r8d
       je       SHORT G_M000_IG39
 
G_M000_IG38:                ;; offset=0x0791
       test     esi, esi
       je       SHORT G_M000_IG39
       mov      r9, r10
       sub      r9, qword ptr [rbp-0x1B8]
       mov      qword ptr [rbp-0xF8], r9
       mov      dword ptr [rbp-0x184], r8d
       mov      r9d, r8d
       shl      r9, 2
       cmp      qword ptr [rbp-0xF8], r9
       jb       G_M000_IG116
       mov      r9d, esi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0xF8]
       mov      r8d, dword ptr [rbp-0x184]
       jb       G_M000_IG116
 
G_M000_IG39:                ;; offset=0x07DF
       cmp      dword ptr [rbp+0x18], 0
       je       SHORT G_M000_IG41
 
G_M000_IG40:                ;; offset=0x07E5
       test     esi, esi
       je       SHORT G_M000_IG41
       mov      r9, r10
       sub      r9, qword ptr [rbp+0x10]
       mov      qword ptr [rbp-0x100], r9
       mov      r9d, dword ptr [rbp+0x18]
       shl      r9, 2
       cmp      qword ptr [rbp-0x100], r9
       jb       G_M000_IG116
       mov      r9d, esi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x100]
       jb       G_M000_IG116
 
G_M000_IG41:                ;; offset=0x0823
       test     r11d, r11d
       je       G_M000_IG53
 
G_M000_IG42:                ;; offset=0x082C
       test     edi, edi
       je       G_M000_IG53
       mov      r9, r13
       sub      r9, r14
       mov      qword ptr [rbp-0x108], r9
       mov      r9d, r11d
       shl      r9, 2
       cmp      qword ptr [rbp-0x108], r9
       jb       G_M000_IG116
       mov      r9d, edi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x108]
       jb       G_M000_IG116
 
G_M000_IG43:                ;; offset=0x086C
       test     esi, esi
       je       G_M000_IG54
       mov      r9, r10
       sub      r9, r14
       mov      qword ptr [rbp-0x110], r9
       mov      r9d, r11d
       shl      r9, 2
       cmp      qword ptr [rbp-0x110], r9
       jb       G_M000_IG116
       mov      r9d, esi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x110]
       jb       G_M000_IG116
 
G_M000_IG44:                ;; offset=0x08AC
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG45
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, r14
       mov      qword ptr [rbp-0x118], r9
       mov      r9d, r11d
       shl      r9, 2
       cmp      qword ptr [rbp-0x118], r9
       jb       G_M000_IG116
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x118]
       jb       G_M000_IG116
 
G_M000_IG45:                ;; offset=0x08EC
       test     edi, edi
       je       G_M000_IG55
 
G_M000_IG46:                ;; offset=0x08F4
       test     esi, esi
       je       G_M000_IG55
       mov      r9, r10
       sub      r9, r13
       mov      qword ptr [rbp-0x120], r9
       mov      r9d, edi
       shl      r9, 2
       cmp      qword ptr [rbp-0x120], r9
       jb       G_M000_IG116
       mov      r9d, esi
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x120]
       jb       G_M000_IG116
 
G_M000_IG47:                ;; offset=0x0934
       cmp      dword ptr [rbp+0x28], 0
       je       SHORT G_M000_IG48
       mov      r9, bword ptr [rbp+0x20]
       sub      r9, r13
       mov      qword ptr [rbp-0x128], r9
       mov      dword ptr [rbp+0x48], edi
       mov      r9d, edi
       shl      r9, 2
       cmp      qword ptr [rbp-0x128], r9
       jb       G_M000_IG116
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x128]
       mov      edi, dword ptr [rbp+0x48]
       jb       G_M000_IG116
 
G_M000_IG48:                ;; offset=0x097A
       test     esi, esi
       je       G_M000_IG56
 
G_M000_IG49:                ;; offset=0x0982
       cmp      dword ptr [rbp+0x28], 0
       je       G_M000_IG117
       mov      r9, bword ptr [rbp+0x20]
       mov      bword ptr [rbp+0x50], r10
       sub      r9, r10
       mov      qword ptr [rbp-0x130], r9
       mov      dword ptr [rbp+0x58], esi
       mov      r9d, esi
       shl      r9, 2
       cmp      qword ptr [rbp-0x130], r9
       jb       G_M000_IG116
       mov      r9d, dword ptr [rbp+0x28]
       shl      r9, 2
       neg      r9
       cmp      r9, qword ptr [rbp-0x130]
       mov      dword ptr [rbp-0x184], r8d
       mov      esi, dword ptr [rbp+0x58]
       mov      r10, bword ptr [rbp+0x50]
       jb       G_M000_IG116
 
G_M000_IG50:                ;; offset=0x09DB
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
       jl       G_M000_IG118
       align    [0 bytes for IG51]
 
G_M000_IG51:                ;; offset=0x0A1C
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
       jne      G_M000_IG119
       add      r8d, 8
       lea      r9d, [r8+0x08]
       mov      ecx, dword ptr [rbp-0x17C]
       cmp      r9d, ecx
       jg       SHORT G_M000_IG57
 
G_M000_IG52:                ;; offset=0x0A64
       mov      dword ptr [rbp-0x17C], ecx
       jmp      SHORT G_M000_IG51
 
G_M000_IG53:                ;; offset=0x0A6C
       test     r11d, r11d
       jne      G_M000_IG43
 
G_M000_IG54:                ;; offset=0x0A75
       test     r11d, r11d
       je       G_M000_IG45
       jmp      G_M000_IG44
 
G_M000_IG55:                ;; offset=0x0A83
       test     edi, edi
       je       G_M000_IG48
       jmp      G_M000_IG47
 
G_M000_IG56:                ;; offset=0x0A90
       mov      dword ptr [rbp-0x184], r8d
       jmp      G_M000_IG50
       align    [0 bytes for IG59]
 
G_M000_IG57:                ;; offset=0x0A9C
       mov      dword ptr [rbp-0x17C], ecx
       cmp      r8d, ecx
       jl       G_M000_IG120
       xor      r8d, r8d
       mov      bword ptr [rbp-0x138], r8
       mov      r8d, 1
 
G_M000_IG58:                ;; offset=0x0ABB
       xor      r9, r9
       mov      bword ptr [rbp-0x138], r9
       test     r8d, r8d
       je       G_M000_IG138
       mov      r8d, edx
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      r9, r9
       test     r8d, r8d
       cmovne   r9, bword ptr [rbp-0x1B0]
       mov      bword ptr [rbp-0x148], r9
       mov      qword ptr [rbp-0x150], r9
       xor      r8d, r8d
       cmp      edx, 8
       jl       G_M000_IG121
       jmp      SHORT G_M000_IG59
 
G_M000_IG59:                ;; offset=0x0B04
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
       jne      G_M000_IG122
       add      r8d, 8
       lea      r9d, [r8+0x08]
       cmp      r9d, edx
       jle      SHORT G_M000_IG59
 
G_M000_IG60:                ;; offset=0x0B46
       mov      dword ptr [rbp-0x180], edx
       cmp      r8d, edx
       jl       G_M000_IG123
       xor      ecx, ecx
       mov      bword ptr [rbp-0x148], rcx
       mov      ecx, 1
 
G_M000_IG61:                ;; offset=0x0B63
       xor      r8, r8
       mov      bword ptr [rbp-0x148], r8
       test     ecx, ecx
       je       G_M000_IG138
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
       jl       SHORT G_M000_IG63
       align    [5 bytes for IG62]
 
G_M000_IG62:                ;; offset=0x0BB0
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
       jne      G_M000_IG125
       add      ecx, 8
       lea      r9d, [rcx+0x08]
       cmp      r9d, dword ptr [rbp-0x184]
       jle      SHORT G_M000_IG62
 
G_M000_IG63:                ;; offset=0x0BEE
       cmp      ecx, dword ptr [rbp-0x184]
       jl       G_M000_IG126
       xor      ecx, ecx
       mov      bword ptr [rbp-0x158], rcx
       mov      ecx, 1
 
G_M000_IG64:                ;; offset=0x0C08
       xor      r8, r8
       mov      bword ptr [rbp-0x158], r8
       test     ecx, ecx
       je       G_M000_IG138
       mov      r8d, dword ptr [rbp+0x18]
       mov      dword ptr [rbp-0x18C], r8d
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
       xor      rcx, rcx
       cmp      dword ptr [rbp-0x18C], 0
       cmovne   rcx, bword ptr [rbp+0x10]
       mov      bword ptr [rbp-0x160], rcx
       xor      r8d, r8d
       cmp      dword ptr [rbp+0x18], 8
       jl       SHORT G_M000_IG66
       align    [4 bytes for IG65]
 
G_M000_IG65:                ;; offset=0x0C50
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
       jne      G_M000_IG127
       add      r8d, 8
       lea      r9d, [r8+0x08]
       cmp      r9d, dword ptr [rbp+0x18]
       jle      SHORT G_M000_IG65
 
G_M000_IG66:                ;; offset=0x0C8C
       cmp      r8d, dword ptr [rbp+0x18]
       jl       G_M000_IG128
       xor      ecx, ecx
       mov      bword ptr [rbp-0x160], rcx
       mov      ecx, 1
 
G_M000_IG67:                ;; offset=0x0CA4
       xor      r8, r8
       mov      bword ptr [rbp-0x160], r8
       test     ecx, ecx
       mov      bword ptr [rbp+0x50], r10
       je       G_M000_IG138
       lea      r8d, [r12+0x01]
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
       jo       G_M000_IG140
       mov      dword ptr [rbp-0x48], r9d
       xor      edx, edx
       mov      r10d, eax
       imul     r10d, r12d
       mov      dword ptr [rbp-0x194], r10d
       cmp      edx, r9d
       jl       G_M000_IG80
       jmp      G_M000_IG129
 
G_M000_IG68:                ;; offset=0x0D15
       mov      r8d, dword ptr [rbp+0x48]
       mov      edx, r8d
       xor      rdi, rdi
       mov      bword ptr [rbp-0x170], rdi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG69:                ;; offset=0x0D2E
       test     edx, edx
       je       SHORT G_M000_IG70
       mov      r13, bword ptr [rbp+0x40]
       mov      rdi, r13
 
G_M000_IG70:                ;; offset=0x0D39
       mov      bword ptr [rbp-0x170], rdi
       xor      edx, edx
       cmp      r8d, 8
       jl       SHORT G_M000_IG72
       align    [0 bytes for IG71]
 
G_M000_IG71:                ;; offset=0x0D48
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
       jne      G_M000_IG134
       add      edx, 8
       lea      esi, [rdx+0x08]
       cmp      esi, r8d
       jle      SHORT G_M000_IG71
       align    [0 bytes for IG72]
 
G_M000_IG72:                ;; offset=0x0D7B
       mov      dword ptr [rbp+0x48], r8d
       cmp      edx, r8d
       jl       G_M000_IG135
       xor      edx, edx
       mov      bword ptr [rbp-0x170], rdx
       mov      edx, 1
 
G_M000_IG73:                ;; offset=0x0D96
       xor      rdi, rdi
       mov      bword ptr [rbp-0x170], rdi
       test     edx, edx
       je       G_M000_IG138
       cmp      dword ptr [rbp+0x80], 16
       jne      G_M000_IG137
       mov      r9d, dword ptr [rbp-0x194]
       mov      dword ptr [rbp-0x174], r9d
       xor      esi, esi
       cmp      esi, ebx
       jl       G_M000_IG103
 
G_M000_IG74:                ;; offset=0x0DCC
       mov      r10d, dword ptr [rbp+0x70]
       mov      r13d, dword ptr [rbp-0x4C]
 
G_M000_IG75:                ;; offset=0x0DD4
       xor      edi, edi
       mov      bword ptr [rbp-0x68], rdi
 
G_M000_IG76:                ;; offset=0x0DDA
       mov      bword ptr [rbp-0x70], rdi
 
G_M000_IG77:                ;; offset=0x0DDE
       mov      bword ptr [rbp-0x78], rdi
 
G_M000_IG78:                ;; offset=0x0DE2
       mov      bword ptr [rbp-0x80], rdi
       add      r13d, 8
       mov      edi, dword ptr [rbp-0x48]
       cmp      r13d, edi
       mov      eax, r10d
       mov      edx, r13d
       mov      r9d, edi
       mov      ecx, dword ptr [rbp-0x44]
       mov      r11d, dword ptr [rbp+0x38]
       mov      r14, bword ptr [rbp+0x30]
       jge      G_M000_IG98
 
G_M000_IG79:                ;; offset=0x0E0A
       mov      esi, dword ptr [rbp+0x58]
       mov      edi, dword ptr [rbp+0x48]
       mov      r13, bword ptr [rbp+0x40]
 
G_M000_IG80:                ;; offset=0x0E14
       mov      r10d, r9d
       sub      r10d, edx
       cmp      r10d, 8
       jl       G_M000_IG130
       mov      r10d, 8
       mov      dword ptr [rbp+0x48], edi
       mov      dword ptr [rbp+0x58], esi
 
G_M000_IG81:                ;; offset=0x0E30
       mov      dword ptr [rbp-0x50], r10d
       mov      dword ptr [rsp], r12d
       mov      dword ptr [rsp+0x08], ecx
       mov      dword ptr [rbp-0x4C], edx
       mov      dword ptr [rsp+0x10], edx
       mov      dword ptr [rsp+0x18], r10d
       mov      rdi, bword ptr [rbp-0x1A8]
       mov      esi, dword ptr [rbp-0x17C]
       mov      rdx, r14
       mov      dword ptr [rbp+0x38], r11d
       mov      ecx, r11d
       mov      r8d, r15d
       mov      dword ptr [rbp+0x70], eax
       mov      r9d, eax
       call     [Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInput(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int,int)]
       mov      eax, dword ptr [rbp+0x38]
       mov      edi, eax
       xor      rsi, rsi
       mov      bword ptr [rbp-0x168], rsi
       vbroadcastss ymm0, dword ptr [reloc @RWD00]
 
G_M000_IG82:                ;; offset=0x0E85
       test     edi, edi
       je       SHORT G_M000_IG83
       mov      rsi, r14
 
G_M000_IG83:                ;; offset=0x0E8C
       mov      bword ptr [rbp-0x168], rsi
       xor      edi, edi
       cmp      eax, 8
       jl       SHORT G_M000_IG85
       align    [6 bytes for IG84]
 
G_M000_IG84:                ;; offset=0x0EA0
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
       jne      G_M000_IG131
       add      edi, 8
       lea      edx, [rdi+0x08]
       cmp      edx, eax
       jle      SHORT G_M000_IG84
       align    [0 bytes for IG85]
 
G_M000_IG85:                ;; offset=0x0ED2
       cmp      edi, eax
       jl       G_M000_IG132
       xor      edi, edi
       mov      bword ptr [rbp-0x168], rdi
       mov      edi, 1
 
G_M000_IG86:                ;; offset=0x0EE8
       xor      rsi, rsi
       mov      bword ptr [rbp-0x168], rsi
       test     edi, edi
       je       G_M000_IG138
       xor      rdi, rdi
       mov      dword ptr [rbp+0x38], eax
       test     eax, eax
       je       SHORT G_M000_IG87
       mov      bword ptr [rbp+0x30], r14
       mov      rdi, r14
       mov      r14, bword ptr [rbp+0x30]
 
G_M000_IG87:                ;; offset=0x0F0D
       mov      bword ptr [rbp-0x68], rdi
       mov      rsi, rdi
       xor      rdx, rdx
       cmp      dword ptr [rbp-0x180], 0
       je       SHORT G_M000_IG88
       mov      r10, bword ptr [rbp-0x1B0]
       mov      rdx, r10
 
G_M000_IG88:                ;; offset=0x0F29
       mov      bword ptr [rbp-0x70], rdx
       mov      rcx, rdx
       xor      r11, r11
       cmp      dword ptr [rbp+0x48], 0
       je       SHORT G_M000_IG89
       mov      bword ptr [rbp+0x40], r13
       mov      r11, r13
       mov      r13, bword ptr [rbp+0x40]
 
G_M000_IG89:                ;; offset=0x0F44
       mov      bword ptr [rbp-0x78], r11
       mov      qword ptr [rbp-0x58], r11
       xor      r10, r10
       cmp      dword ptr [rbp+0x58], 0
       je       SHORT G_M000_IG90
       mov      r10, bword ptr [rbp+0x50]
 
G_M000_IG90:                ;; offset=0x0F59
       mov      bword ptr [rbp-0x80], r10
       mov      qword ptr [rbp-0x60], r10
       cmp      dword ptr [rbp+0x80], 16
       jne      G_M000_IG133
       xor      edi, edi
       movsxd   rdx, ebx
       shl      rdx, 2
       jmp      G_M000_IG97
       align    [0 bytes for IG93]
 
G_M000_IG91:                ;; offset=0x0F7C
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       mov      r13d, edi
       imul     r13d, r15d
       mov      r14d, r13d
       imul     r14d, ebx
       movsxd   r14, r14d
       shl      r14, 2
       add      r14, rcx
       movsxd   r10, r9d
       lea      r10, [r14+4*r10]
       shl      r13d, 3
       movsxd   r14, r13d
       lea      r14, [rsi+4*r14]
       test     r15d, r15d
       jle      SHORT G_M000_IG94
 
G_M000_IG92:                ;; offset=0x0FCB
       mov      r13d, r15d
 
G_M000_IG93:                ;; offset=0x0FCE
       vmovups  zmm8, zmmword ptr [r10]
       vfmadd231ps zmm0, zmm8, dword ptr [r14] {1to16}
       vfmadd231ps zmm1, zmm8, dword ptr [r14+0x04] {1to16}
       vfmadd231ps zmm2, zmm8, dword ptr [r14+0x08] {1to16}
       vfmadd231ps zmm3, zmm8, dword ptr [r14+0x0C] {1to16}
       vfmadd231ps zmm4, zmm8, dword ptr [r14+0x10] {1to16}
       vfmadd231ps zmm5, zmm8, dword ptr [r14+0x14] {1to16}
       vfmadd231ps zmm6, zmm8, dword ptr [r14+0x18] {1to16}
       vfmadd231ps zmm7, zmm8, dword ptr [r14+0x1C] {1to16}
       add      r10, rdx
       add      r14, 32
       dec      r13d
       jne      SHORT G_M000_IG93
 
G_M000_IG94:                ;; offset=0x1017
       mov      r10d, edi
       imul     r10d, ebx
       add      r10d, r9d
       shl      r10d, 3
       movsxd   r10, r10d
       lea      r10, [r11+4*r10]
       vmovups  zmmword ptr [r10], zmm0
       vmovups  zmmword ptr [r10+0x40], zmm1
       vmovups  zmmword ptr [r10+0x80], zmm2
       vmovups  zmmword ptr [r10+0xC0], zmm3
       vmovups  zmmword ptr [r10+0x100], zmm4
       vmovups  zmmword ptr [r10+0x140], zmm5
       vmovups  zmmword ptr [r10+0x180], zmm6
       vmovups  zmmword ptr [r10+0x1C0], zmm7
       add      r9d, 16
       cmp      r9d, ebx
       jl       G_M000_IG91
 
G_M000_IG95:                ;; offset=0x1070
       inc      edi
       cmp      edi, 16
       jge      G_M000_IG68
 
G_M000_IG96:                ;; offset=0x107B
       mov      r13, bword ptr [rbp+0x40]
       mov      r14, bword ptr [rbp+0x30]
 
G_M000_IG97:                ;; offset=0x1083
       xor      r9d, r9d
       cmp      r9d, ebx
       mov      bword ptr [rbp+0x30], r14
       mov      bword ptr [rbp+0x40], r13
       jl       G_M000_IG91
       jmp      SHORT G_M000_IG95
 
G_M000_IG98:                ;; offset=0x1099
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:Finite(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG138
       mov      rdi, bword ptr [rbp+0x50]
       mov      esi, dword ptr [rbp+0x58]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG138
       mov      rdi, bword ptr [rbp-0x1B8]
       mov      esi, dword ptr [rbp-0x184]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG138
       mov      rdi, bword ptr [rbp+0x10]
       mov      esi, dword ptr [rbp+0x18]
       call     [Lokad.Onnx.ConvBlockedSpatial:EpilogueRange(System.ReadOnlySpan`1[float]):bool]
       test     eax, eax
       je       G_M000_IG138
       mov      r15, bword ptr [rbp+0x10]
       mov      bword ptr [rsp], r15
       mov      r14d, dword ptr [rbp+0x18]
       mov      dword ptr [rsp+0x08], r14d
       mov      dword ptr [rsp+0x10], ebx
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
 
G_M000_IG99:                ;; offset=0x114E
       vzeroupper 
       add      rsp, 456
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG100:                ;; offset=0x1163
       mov      eax, dword ptr [rbp-0x174]
       mov      edi, dword ptr [rbp-0x50]
       mov      r10d, dword ptr [rbp+0x70]
       mov      r13d, dword ptr [rbp-0x4C]
       mov      r14, qword ptr [rbp-0x60]
 
G_M000_IG101:                ;; offset=0x1178
       add      esi, 16
       cmp      esi, ebx
       jge      G_M000_IG75
 
G_M000_IG102:                ;; offset=0x1183
       mov      dword ptr [rbp-0x4C], r13d
       mov      dword ptr [rbp+0x70], r10d
 
G_M000_IG103:                ;; offset=0x118B
       xor      ecx, ecx
       cmp      ecx, dword ptr [rbp-0x50]
       jge      SHORT G_M000_IG100
       align    [0 bytes for IG104]
 
G_M000_IG104:                ;; offset=0x1192
       lea      r9d, [8*rsi]
       movsxd   r9, r9d
       shl      r9, 2
       add      r9, r11
       mov      r10d, ecx
       shl      r10d, 4
       movsxd   r10, r10d
       lea      r9, [r9+4*r10]
       lea      r10d, [8*rbx]
       mov      r13d, dword ptr [rbp-0x4C]
       lea      r14d, [rcx+r13]
       mov      eax, r14d
       cdq      
       idiv     edx:eax, dword ptr [rbp-0x44]
       lea      edx, [rax+rax]
       mov      dword ptr [rbp-0x178], edx
       mov      eax, r14d
       cdq      
       idiv     edx:eax, dword ptr [rbp-0x44]
       add      edx, edx
       vmovups  zmm0, zmmword ptr [r9]
       lea      eax, [4*r10]
       cdqe     
       vmovups  zmm1, zmmword ptr [r9+4*rax]
       vaddps   zmm0, zmm0, zmm1
       lea      eax, [8*r10]
       cdqe     
       vmovups  zmm2, zmmword ptr [r9+4*rax]
       vaddps   zmm0, zmm0, zmm2
       vsubps   zmm1, zmm1, zmm2
       lea      eax, [r10+2*r10]
       lea      r14d, [4*rax]
       movsxd   r14, r14d
       vsubps   zmm1, zmm1, zmmword ptr [r9+4*r14]
       movsxd   r14, r10d
       vmovups  zmm2, zmmword ptr [r9+4*r14]
       lea      r14d, [r10+4*r10]
       movsxd   rdi, r14d
       vmovups  zmm3, zmmword ptr [r9+4*rdi]
       vaddps   zmm2, zmm2, zmm3
       lea      edi, [r10+8*r10]
       movsxd   rdi, edi
       vmovups  zmm4, zmmword ptr [r9+4*rdi]
       vaddps   zmm2, zmm2, zmm4
       vsubps   zmm3, zmm3, zmm4
       imul     edi, r10d, 13
       movsxd   rdi, edi
       vsubps   zmm3, zmm3, zmmword ptr [r9+4*rdi]
       lea      edi, [r10+r10]
       movsxd   rdi, edi
       vmovups  zmm4, zmmword ptr [r9+4*rdi]
       lea      edi, [rax+rax]
       movsxd   rdi, edi
       vmovups  zmm5, zmmword ptr [r9+4*rdi]
       vaddps   zmm4, zmm4, zmm5
       add      r14d, r14d
       movsxd   rdi, r14d
       vmovups  zmm6, zmmword ptr [r9+4*rdi]
       vaddps   zmm4, zmm6, zmm4
       vsubps   zmm5, zmm5, zmm6
       imul     edi, r10d, 14
       movsxd   rdi, edi
       vsubps   zmm5, zmm5, zmmword ptr [r9+4*rdi]
       movsxd   rdi, eax
       vmovups  zmm6, zmmword ptr [r9+4*rdi]
       lea      edi, [8*r10]
 
G_M000_IG105:                ;; offset=0x12CB
       sub      edi, r10d
       movsxd   rdi, edi
       vmovups  zmm7, zmmword ptr [r9+4*rdi]
       vaddps   zmm6, zmm6, zmm7
       imul     edi, r10d, 11
       movsxd   rdi, edi
       vmovups  zmm8, zmmword ptr [r9+4*rdi]
       vaddps   zmm6, zmm6, zmm8
       vsubps   zmm7, zmm7, zmm8
       mov      edi, r10d
       shl      edi, 4
       sub      edi, r10d
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
       mov      edi, dword ptr [rbp-0x178]
       mov      r9d, edi
       imul     r9d, r12d
       add      r9d, edx
       shl      r9d, 4
       movsxd   r9, r9d
       mov      eax, dword ptr [rbp-0x174]
       mov      r10d, esi
       imul     r10d, eax
       movsxd   r10, r10d
       shl      r10, 2
       mov      r14, qword ptr [rbp-0x60]
       add      r10, r14
       lea      r9, [r10+4*r9]
       vmovups  zmmword ptr [r9], zmm0
       lea      r10d, [rdx+0x01]
       cmp      r10d, r12d
       jge      SHORT G_M000_IG107
 
G_M000_IG106:                ;; offset=0x1380
       vmovups  zmmword ptr [r9+0x40], zmm2
 
G_M000_IG107:                ;; offset=0x1387
       inc      edi
       mov      r10d, dword ptr [rbp+0x70]
       cmp      edi, r10d
       jge      SHORT G_M000_IG110
 
G_M000_IG108:                ;; offset=0x1392
       mov      edi, r12d
       shl      edi, 4
       movsxd   rdi, edi
       vmovups  zmmword ptr [r9+4*rdi], zmm1
       inc      edx
       cmp      edx, r12d
       jge      SHORT G_M000_IG110
 
G_M000_IG109:                ;; offset=0x13A9
       mov      edx, dword ptr [rbp-0x19C]
       mov      edi, edx
       shl      edi, 4
       movsxd   rdi, edi
       vmovups  zmmword ptr [r9+4*rdi], zmm3
 
G_M000_IG110:                ;; offset=0x13BE
       inc      ecx
       mov      edi, dword ptr [rbp-0x50]
       cmp      ecx, edi
       jge      G_M000_IG101
 
G_M000_IG111:                ;; offset=0x13CB
       mov      dword ptr [rbp-0x4C], r13d
       mov      dword ptr [rbp+0x70], r10d
       jmp      G_M000_IG104
 
G_M000_IG112:                ;; offset=0x13D8
       mov      r9d, 1
       jmp      G_M000_IG04
 
G_M000_IG113:                ;; offset=0x13E3
       mov      rdi, 0x755073E9CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0xC72
       mov      rsi, 0x755073F51A30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG114:                ;; offset=0x141F
       mov      rdi, 0x755073E9CD28
       call     CORINFO_HELP_NEWSFAST
       mov      rbx, rax
       mov      edi, 0xC0A
       mov      rsi, 0x755073F51A30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, rbx
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, rbx
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG115:                ;; offset=0x145B
       call     [System.ThrowHelper:ThrowArgumentOutOfRangeException()]
       int3     
 
G_M000_IG116:                ;; offset=0x1462
       mov      rdi, 0x755073E9CD28
       call     CORINFO_HELP_NEWSFAST
       mov      r12, rax
       mov      edi, 0xC38
       mov      rsi, 0x755073F51A30
       call     [CORINFO_HELP_STRCNS]
       mov      rsi, rax
       mov      rdi, r12
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, r12
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG117:                ;; offset=0x149E
       mov      dword ptr [rbp-0x184], r8d
       jmp      G_M000_IG50
 
G_M000_IG118:                ;; offset=0x14AA
       mov      ecx, dword ptr [rbp-0x17C]
       jmp      G_M000_IG57
 
G_M000_IG119:                ;; offset=0x14B5
       xor      r8d, r8d
       jmp      G_M000_IG58
 
G_M000_IG120:                ;; offset=0x14BD
       movsxd   r9, r8d
       mov      rcx, qword ptr [rbp-0x140]
       mov      r9d, dword ptr [rcx+4*r9]
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, 0x7F800000
       mov      dword ptr [rbp-0x1C0], r9d
       mov      r9d, dword ptr [rbp-0x1BC]
       andn     r9d, r9d, dword ptr [rbp-0x1C0]
       je       SHORT G_M000_IG119
       inc      r8d
       mov      ecx, dword ptr [rbp-0x17C]
       jmp      G_M000_IG57
 
G_M000_IG121:                ;; offset=0x14FF
       mov      rcx, qword ptr [rbp-0x150]
       jmp      G_M000_IG60
 
G_M000_IG122:                ;; offset=0x150B
       xor      ecx, ecx
       mov      dword ptr [rbp-0x180], edx
       jmp      G_M000_IG61
 
G_M000_IG123:                ;; offset=0x1518
       movsxd   r9, r8d
       mov      r9d, dword ptr [rcx+4*r9]
       mov      dword ptr [rbp-0x1C0], r9d
       mov      r9d, 0x7F800000
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, dword ptr [rbp-0x1C0]
       andn     r9d, r9d, dword ptr [rbp-0x1BC]
       je       SHORT G_M000_IG124
       inc      r8d
       mov      edx, dword ptr [rbp-0x180]
       jmp      G_M000_IG60
 
G_M000_IG124:                ;; offset=0x1553
       mov      edx, dword ptr [rbp-0x180]
       jmp      SHORT G_M000_IG122
 
G_M000_IG125:                ;; offset=0x155B
       xor      ecx, ecx
       jmp      G_M000_IG64
 
G_M000_IG126:                ;; offset=0x1562
       movsxd   r9, ecx
       mov      r9d, dword ptr [r8+4*r9]
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, 0x7F800000
       mov      dword ptr [rbp-0x1C0], r9d
       mov      r9d, dword ptr [rbp-0x1BC]
       andn     r9d, r9d, dword ptr [rbp-0x1C0]
       je       SHORT G_M000_IG125
       inc      ecx
       jmp      G_M000_IG63
 
G_M000_IG127:                ;; offset=0x1596
       xor      ecx, ecx
       jmp      G_M000_IG67
 
G_M000_IG128:                ;; offset=0x159D
       movsxd   r9, r8d
       mov      r9d, dword ptr [rcx+4*r9]
       mov      dword ptr [rbp-0x1C0], r9d
       mov      r9d, 0x7F800000
       mov      dword ptr [rbp-0x1BC], r9d
       mov      r9d, dword ptr [rbp-0x1C0]
       andn     r9d, r9d, dword ptr [rbp-0x1BC]
       je       SHORT G_M000_IG127
       inc      r8d
       jmp      G_M000_IG66
 
G_M000_IG129:                ;; offset=0x15D2
       mov      dword ptr [rbp+0x58], esi
       jmp      G_M000_IG98
 
G_M000_IG130:                ;; offset=0x15DA
       mov      dword ptr [rbp+0x48], edi
       mov      dword ptr [rbp+0x58], esi
       jmp      G_M000_IG81
 
G_M000_IG131:                ;; offset=0x15E5
       xor      edi, edi
       jmp      G_M000_IG86
 
G_M000_IG132:                ;; offset=0x15EC
       movsxd   rdx, edi
       mov      edx, dword ptr [rsi+4*rdx]
       mov      ecx, 0x7F800000
       andn     edx, edx, ecx
       je       SHORT G_M000_IG131
       inc      edi
       jmp      G_M000_IG85
 
G_M000_IG133:                ;; offset=0x1605
       mov      rsi, rdx
       mov      rdx, r11
       mov      ecx, r15d
       mov      r8d, ebx
       call     [Lokad.Onnx.ConvBlockedSpatial:MultiplyWinograd256(ptr,ptr,ptr,int,int)]
       mov      bword ptr [rbp+0x30], r14
       mov      bword ptr [rbp+0x40], r13
       mov      r11, qword ptr [rbp-0x58]
       jmp      G_M000_IG68
 
G_M000_IG134:                ;; offset=0x1628
       xor      edx, edx
       mov      dword ptr [rbp+0x48], r8d
       jmp      G_M000_IG73
 
G_M000_IG135:                ;; offset=0x1633
       movsxd   rsi, edx
       mov      esi, dword ptr [rdi+4*rsi]
       mov      ecx, 0x7F800000
       andn     esi, esi, ecx
       je       SHORT G_M000_IG136
       inc      edx
       mov      r8d, dword ptr [rbp+0x48]
       jmp      G_M000_IG72
 
G_M000_IG136:                ;; offset=0x1650
       mov      r8d, dword ptr [rbp+0x48]
       jmp      SHORT G_M000_IG134
 
G_M000_IG137:                ;; offset=0x1656
       mov      r9d, dword ptr [rbp-0x4C]
       mov      dword ptr [rsp], r9d
       mov      edi, dword ptr [rbp-0x50]
       mov      dword ptr [rsp+0x08], edi
       mov      rdi, r11
       mov      rsi, qword ptr [rbp-0x60]
       mov      edx, ebx
       mov      ecx, dword ptr [rbp+0x70]
       mov      r8d, r12d
       mov      r9d, dword ptr [rbp-0x44]
       call     [Lokad.Onnx.ConvBlockedSpatial:OutputWinograd256(ptr,ptr,int,int,int,int,int,int)]
       mov      r10d, dword ptr [rbp+0x70]
       mov      r13d, dword ptr [rbp-0x4C]
       jmp      G_M000_IG75
 
G_M000_IG138:                ;; offset=0x168B
       xor      eax, eax
 
G_M000_IG139:                ;; offset=0x168D
       vzeroupper 
       add      rsp, 456
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
G_M000_IG140:                ;; offset=0x16A2
       call     CORINFO_HELP_OVERFLOW
       int3     
 
RWD00  	dd	7F800000h		;       inf

; Total bytes of code 5800

