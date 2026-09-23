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
       mov      rdi, 0x7C0BEB524998
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
       mov      rdi, 0x7C0BEB52499C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG04:                ;; offset=0x0128
       cmp      dword ptr [rbp+0x18], 0
       je       SHORT G_M000_IG05
       mov      eax, dword ptr [rbp+0x18]
       cmp      eax, dword ptr [rbp-0x70]
       jne      G_M000_IG10
       mov      rdi, 0x7C0BEB5249A0
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
       mov      rdi, 0x7C0BEB5249A4
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG07:                ;; offset=0x0178
       mov      rdi, 0x7C0BEAA9CD28
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0xC8], rax
       mov      edi, 0xC0A
       mov      rsi, 0x7C0BEAB51A30
       call     [CORINFO_HELP_STRCNS]
       mov      gword ptr [rbp-0x428], rax
       mov      rsi, gword ptr [rbp-0x428]
       mov      rdi, gword ptr [rbp-0xC8]
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, gword ptr [rbp-0xC8]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG08:                ;; offset=0x01CB
       mov      rdi, 0x7C0BEB5249A8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG07
 
G_M000_IG09:                ;; offset=0x01DC
       mov      rdi, 0x7C0BEB5249AC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG07
 
G_M000_IG10:                ;; offset=0x01ED
       mov      rdi, 0x7C0BEB5249B0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG11:                ;; offset=0x0201
       mov      rdi, 0x7C0BEB5249B4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG12:                ;; offset=0x0215
       mov      rdi, 0x7C0BEB5249B8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG07
 
G_M000_IG13:                ;; offset=0x0229
       mov      rdi, 0x7C0BEB5249BC
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
       mov      rdi, 0x7C0BEB5249C0
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG55:                ;; offset=0x0A46
       mov      rdi, 0x7C0BEAA9CD28
       call     CORINFO_HELP_NEWSFAST
       mov      gword ptr [rbp-0x120], rax
       mov      edi, 0xC38
       mov      rsi, 0x7C0BEAB51A30
       call     [CORINFO_HELP_STRCNS]
       mov      gword ptr [rbp-0x430], rax
       mov      rsi, gword ptr [rbp-0x430]
       mov      rdi, gword ptr [rbp-0x120]
       call     [System.ArgumentException:.ctor(System.String):this]
       mov      rdi, gword ptr [rbp-0x120]
       call     CORINFO_HELP_THROW
       int3     
 
G_M000_IG56:                ;; offset=0x0A99
       mov      rdi, 0x7C0BEB5249C4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG55
 
G_M000_IG57:                ;; offset=0x0AAA
       mov      rdi, 0x7C0BEB5249C8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG55
 
G_M000_IG58:                ;; offset=0x0ABB
       mov      rdi, 0x7C0BEB5249CC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG59:                ;; offset=0x0ACF
       mov      rdi, 0x7C0BEB5249D0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG60:                ;; offset=0x0AE3
       mov      rdi, 0x7C0BEB5249D4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG61:                ;; offset=0x0AF7
       mov      rdi, 0x7C0BEB5249D8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG62:                ;; offset=0x0B0B
       mov      rdi, 0x7C0BEB5249DC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG63:                ;; offset=0x0B1F
       mov      rdi, 0x7C0BEB5249E0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG64:                ;; offset=0x0B33
       mov      rdi, 0x7C0BEB5249E4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG65:                ;; offset=0x0B47
       mov      rdi, 0x7C0BEB5249E8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG66:                ;; offset=0x0B5B
       mov      rdi, 0x7C0BEB5249EC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG67:                ;; offset=0x0B6F
       mov      rdi, 0x7C0BEB5249F0
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG68:                ;; offset=0x0B83
       mov      rdi, 0x7C0BEB5249F4
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG69:                ;; offset=0x0B97
       mov      rdi, 0x7C0BEB5249F8
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG70:                ;; offset=0x0BAB
       mov      rdi, 0x7C0BEB5249FC
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG71:                ;; offset=0x0BBF
       mov      rdi, 0x7C0BEB524A00
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG72:                ;; offset=0x0BD3
       mov      rdi, 0x7C0BEB524A04
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG73:                ;; offset=0x0BE7
       mov      rdi, 0x7C0BEB524A08
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG74:                ;; offset=0x0BFB
       mov      rdi, 0x7C0BEB524A0C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG75:                ;; offset=0x0C0F
       mov      rdi, 0x7C0BEB524A10
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG55
 
G_M000_IG76:                ;; offset=0x0C23
       cmp      dword ptr [rbp+0x80], 16
       jne      SHORT G_M000_IG77
       mov      rdi, 0x7C0BEB524A14
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG78
 
G_M000_IG77:                ;; offset=0x0C3D
       cmp      dword ptr [rbp+0x80], 8
       jne      SHORT G_M000_IG79
       mov      rdi, 0x7C0BEB524A18
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG79
 
G_M000_IG78:                ;; offset=0x0C57
       mov      rdi, 0x7C0BEB39DBD8
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
       mov      rdi, 0x7C0BEB524A1C
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG81:                ;; offset=0x0CDE
       mov      rdi, 0x7C0BEB524A20
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      G_M000_IG97
 
G_M000_IG82:                ;; offset=0x0CF2
       mov      rdi, 0x7C0BEB524A24
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG81
 
G_M000_IG83:                ;; offset=0x0D03
       mov      rdi, 0x7C0BEB524A28
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
       mov      rdi, 0x7C0BEB524A2C
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
       mov      rdi, 0x7C0BEB524A30
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
       mov      rdi, 0x7C0BEB524A34
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
       mov      rdi, 0x7C0BEB524A38
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
       mov      rdi, 0x7C0BEB524A3C
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
       mov      rdi, 0x7C0BEB524A40
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG96:                ;; offset=0x10E1
       mov      rdi, 0x7C0BEB524A44
       call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG97:                ;; offset=0x10F0
       xor      eax, eax
 
G_M000_IG98:                ;; offset=0x10F2
       add      rsp, 0x490
       pop      rbp
       ret      
 
G_M000_IG99:                ;; offset=0x10FB
       mov      rdi, 0x7C0BEB524A48
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG96
 
G_M000_IG100:                ;; offset=0x110C
       mov      rdi, 0x7C0BEB524A4C
       call     CORINFO_HELP_COUNTPROFILE32
       jmp      SHORT G_M000_IG96
 
G_M000_IG101:                ;; offset=0x111D
       mov      rdi, 0x7C0BEB524A50
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

