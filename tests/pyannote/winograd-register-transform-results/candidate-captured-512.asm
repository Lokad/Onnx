; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x950
       lea      rbp, [rsp+0x950]
       vxorps   xmm8, xmm8, xmm8
       mov      rax, -0x660
       vmovdqa  xmmword ptr [rbp+rax-0x50], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      bword ptr [rbp-0x38], rdi
       mov      qword ptr [rbp-0x30], rsi
       mov      bword ptr [rbp-0x48], rdx
       mov      qword ptr [rbp-0x40], rcx
       mov      dword ptr [rbp-0x4C], r8d
       mov      dword ptr [rbp-0x50], r9d
 
G_M000_IG02:                ;; offset=0x004F
       mov      dword ptr [rbp-0x938], 0x3E8
       vmovups  ymm0, ymmword ptr [reloc @RWD00]
       vmovups  ymmword ptr [rbp-0x70], ymm0
       lea      rdi, [rbp-0x38]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0x88], rax
       mov      rax, bword ptr [rbp-0x88]
       mov      qword ptr [rbp-0x940], rax
       mov      rax, qword ptr [rbp-0x940]
       mov      qword ptr [rbp-0x78], rax
       lea      rdi, [rbp-0x48]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0x90], rax
       mov      rax, bword ptr [rbp-0x90]
       mov      qword ptr [rbp-0x948], rax
       mov      rax, qword ptr [rbp-0x948]
       mov      qword ptr [rbp-0x80], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x94], eax
       jmp      G_M000_IG10
 
G_M000_IG03:                ;; offset=0x00C7
       mov      rdi, 0x76079FF51990
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x94]
       imul     eax, dword ptr [rbp-0x50]
       imul     eax, dword ptr [rbp+0x10]
       cdqe     
       mov      rcx, qword ptr [rbp-0x78]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0xA0], rax
       mov      eax, dword ptr [rbp+0x18]
       imul     eax, dword ptr [rbp+0x10]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0xA0]
       movsxd   rcx, dword ptr [rbp+0x20]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x6F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x6F0]
       vmovups  ymmword ptr [rbp-0xD0], ymm0
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  ymm0, ymmword ptr [rax+0x20]
       vmovups  ymmword ptr [rbp-0x710], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x710]
       vmovups  ymmword ptr [rbp-0xF0], ymm0
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  ymm0, ymmword ptr [rax+0x08]
       vmovups  ymmword ptr [rbp-0x730], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x730]
       vmovups  ymmword ptr [rbp-0x110], ymm0
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  ymm0, ymmword ptr [rax+0x28]
       vmovups  ymmword ptr [rbp-0x750], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x750]
       vmovups  ymmword ptr [rbp-0x130], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0xD0]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0xF0], 32
       vmovups  ymmword ptr [rbp-0x150], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0xD0]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0xF0], 49
       vmovups  ymmword ptr [rbp-0x170], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x110]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x130], 32
       vmovups  ymmword ptr [rbp-0x190], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x110]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x130], 49
       vmovups  ymmword ptr [rbp-0x1B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x150]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x1D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x170]
 
G_M000_IG04:                ;; offset=0x0247
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x1F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x190]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x170]
       vmovups  ymmword ptr [rbp-0x210], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x170]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x1B0]
       vmovups  ymmword ptr [rbp-0x230], ymm0
       mov      eax, dword ptr [rbp+0x18]
       add      eax, 2
       imul     eax, dword ptr [rbp+0x10]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0xA0]
       movsxd   rcx, dword ptr [rbp+0x20]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x758], rax
       mov      rax, qword ptr [rbp-0x758]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x790], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x790]
       vmovups  ymmword ptr [rbp-0x250], ymm0
       mov      rax, qword ptr [rbp-0x758]
       vmovups  ymm0, ymmword ptr [rax+0x20]
       vmovups  ymmword ptr [rbp-0x7B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x7B0]
       vmovups  ymmword ptr [rbp-0x270], ymm0
       mov      rax, qword ptr [rbp-0x758]
       vmovups  ymm0, ymmword ptr [rax+0x08]
       vmovups  ymmword ptr [rbp-0x7D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x7D0]
       vmovups  ymmword ptr [rbp-0x290], ymm0
       mov      rax, qword ptr [rbp-0x758]
       vmovups  ymm0, ymmword ptr [rax+0x28]
       vmovups  ymmword ptr [rbp-0x7F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x7F0]
       vmovups  ymmword ptr [rbp-0x2B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x250]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x270], 32
       vmovups  ymmword ptr [rbp-0x2D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x250]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x270], 49
       vmovups  ymmword ptr [rbp-0x2F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x290]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x2B0], 32
       vmovups  ymmword ptr [rbp-0x310], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x290]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x2B0], 49
       vmovups  ymmword ptr [rbp-0x330], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x2D0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x310]
       vmovups  ymmword ptr [rbp-0x350], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x2F0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x310]
 
G_M000_IG05:                ;; offset=0x03E4
       vmovups  ymmword ptr [rbp-0x370], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x310]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x2F0]
       vmovups  ymmword ptr [rbp-0x390], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x2F0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x330]
       vmovups  ymmword ptr [rbp-0x3B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x1D0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x350]
       mov      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x1F0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x370]
       mov      eax, dword ptr [rbp-0x4C]
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x210]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x390]
       mov      eax, dword ptr [rbp-0x4C]
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+2*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x230]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x3B0]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+2*rax]
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp+0x18]
       inc      eax
       imul     eax, dword ptr [rbp+0x10]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0xA0]
       movsxd   rcx, dword ptr [rbp+0x20]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x7F8], rax
       mov      rax, qword ptr [rbp-0x7F8]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x830], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x830]
       vmovups  ymmword ptr [rbp-0x3D0], ymm0
       mov      rax, qword ptr [rbp-0x7F8]
       vmovups  ymm0, ymmword ptr [rax+0x20]
       vmovups  ymmword ptr [rbp-0x850], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x850]
       vmovups  ymmword ptr [rbp-0x3F0], ymm0
 
G_M000_IG06:                ;; offset=0x0533
       mov      rax, qword ptr [rbp-0x7F8]
       vmovups  ymm0, ymmword ptr [rax+0x08]
       vmovups  ymmword ptr [rbp-0x870], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x870]
       vmovups  ymmword ptr [rbp-0x410], ymm0
       mov      rax, qword ptr [rbp-0x7F8]
       vmovups  ymm0, ymmword ptr [rax+0x28]
       vmovups  ymmword ptr [rbp-0x890], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x890]
       vmovups  ymmword ptr [rbp-0x430], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x3D0]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x3F0], 32
       vmovups  ymmword ptr [rbp-0x450], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x3D0]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x3F0], 49
       vmovups  ymmword ptr [rbp-0x470], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x410]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x430], 32
       vmovups  ymmword ptr [rbp-0x490], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x410]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x430], 49
       vmovups  ymmword ptr [rbp-0x4B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x450]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x490]
       vmovups  ymmword ptr [rbp-0x4D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x470]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x490]
       vmovups  ymmword ptr [rbp-0x4F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x490]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x470]
       vmovups  ymmword ptr [rbp-0x510], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x470]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x4B0]
       vmovups  ymmword ptr [rbp-0x530], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x4D0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x350]
       mov      eax, dword ptr [rbp-0x4C]
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+4*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x4F0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x370]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+4*rax]
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x510]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x390]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+2*rax]
 
G_M000_IG07:                ;; offset=0x06B9
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+2*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x530]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x3B0]
       imul     eax, dword ptr [rbp-0x4C], 7
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x350]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x4D0]
       mov      eax, dword ptr [rbp-0x4C]
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+8*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x370]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x4F0]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+8*rax]
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x390]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x510]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+4*rax]
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+2*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x3B0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x530]
       imul     eax, dword ptr [rbp-0x4C], 11
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp+0x18]
       add      eax, 3
       imul     eax, dword ptr [rbp+0x10]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0xA0]
       movsxd   rcx, dword ptr [rbp+0x20]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x898], rax
       mov      rax, qword ptr [rbp-0x898]
       vmovups  ymm0, ymmword ptr [rax]
 
G_M000_IG08:                ;; offset=0x07D2
       vmovups  ymmword ptr [rbp-0x8D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x8D0]
       vmovups  ymmword ptr [rbp-0x550], ymm0
       mov      rax, qword ptr [rbp-0x898]
       vmovups  ymm0, ymmword ptr [rax+0x20]
       vmovups  ymmword ptr [rbp-0x8F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x8F0]
       vmovups  ymmword ptr [rbp-0x570], ymm0
       mov      rax, qword ptr [rbp-0x898]
       vmovups  ymm0, ymmword ptr [rax+0x08]
       vmovups  ymmword ptr [rbp-0x910], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x910]
       vmovups  ymmword ptr [rbp-0x590], ymm0
       mov      rax, qword ptr [rbp-0x898]
       vmovups  ymm0, ymmword ptr [rax+0x28]
       vmovups  ymmword ptr [rbp-0x930], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x930]
       vmovups  ymmword ptr [rbp-0x5B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x550]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x570], 32
       vmovups  ymmword ptr [rbp-0x5D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x550]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x570], 49
       vmovups  ymmword ptr [rbp-0x5F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x590]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x5B0], 32
       vmovups  ymmword ptr [rbp-0x610], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x590]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x5B0], 49
       vmovups  ymmword ptr [rbp-0x630], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x5D0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x610]
       vmovups  ymmword ptr [rbp-0x650], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x5F0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x610]
       vmovups  ymmword ptr [rbp-0x670], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x610]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x5F0]
       vmovups  ymmword ptr [rbp-0x690], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x5F0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x630]
       vmovups  ymmword ptr [rbp-0x6B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x4D0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x650]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+2*rax]
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+4*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x4F0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x670]
 
G_M000_IG09:                ;; offset=0x0973
       imul     eax, dword ptr [rbp-0x4C], 13
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x510]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x690]
       imul     eax, dword ptr [rbp-0x4C], 14
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x530]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x6B0]
       imul     eax, dword ptr [rbp-0x4C], 15
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp-0x94]
       inc      eax
       mov      dword ptr [rbp-0x94], eax
 
G_M000_IG10:                ;; offset=0x09E9
       mov      eax, dword ptr [rbp-0x938]
       dec      eax
       mov      dword ptr [rbp-0x938], eax
       cmp      dword ptr [rbp-0x938], 0
       jg       SHORT G_M000_IG12
 
G_M000_IG11:                ;; offset=0x0A00
       lea      rdi, [rbp-0x938]
       mov      esi, 0x4D9
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG12:                ;; offset=0x0A11
       mov      eax, dword ptr [rbp-0x94]
       cmp      eax, dword ptr [rbp-0x4C]
       jl       G_M000_IG03
       mov      rdi, 0x76079FF51994
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
       mov      bword ptr [rbp-0x88], rax
       xor      eax, eax
       mov      bword ptr [rbp-0x90], rax
 
G_M000_IG13:                ;; offset=0x0A41
       vzeroupper 
       add      rsp, 0x950
       pop      rbp
       ret      
 
RWD00  	dq	0000000200000000h, 0000000600000004h, 0000000300000001h, 0000000700000005h

; Total bytes of code 2637

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       sub      rsp, 0x950
       lea      rbp, [rsp+0x950]
       vxorps   xmm8, xmm8, xmm8
       mov      rax, -0x660
       vmovdqa  xmmword ptr [rbp+rax-0x50], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       add      rax, 48
       jne      SHORT  -5 instr
       mov      bword ptr [rbp-0x38], rdi
       mov      qword ptr [rbp-0x30], rsi
       mov      bword ptr [rbp-0x48], rdx
       mov      qword ptr [rbp-0x40], rcx
       mov      dword ptr [rbp-0x4C], r8d
       mov      dword ptr [rbp-0x50], r9d
 
G_M000_IG02:                ;; offset=0x004F
       mov      dword ptr [rbp-0x938], 0x3E8
       vmovups  ymm0, ymmword ptr [reloc @RWD00]
       vmovups  ymmword ptr [rbp-0x70], ymm0
       lea      rdi, [rbp-0x38]
       call     [System.ReadOnlySpan`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0x88], rax
       mov      rax, bword ptr [rbp-0x88]
       mov      qword ptr [rbp-0x940], rax
       mov      rax, qword ptr [rbp-0x940]
       mov      qword ptr [rbp-0x78], rax
       lea      rdi, [rbp-0x48]
       call     [System.Span`1[float]:GetPinnableReference():byref:this]
       mov      bword ptr [rbp-0x90], rax
       mov      rax, bword ptr [rbp-0x90]
       mov      qword ptr [rbp-0x948], rax
       mov      rax, qword ptr [rbp-0x948]
       mov      qword ptr [rbp-0x80], rax
       xor      eax, eax
       mov      dword ptr [rbp-0x94], eax
       jmp      G_M000_IG10
 
G_M000_IG03:                ;; offset=0x00C7
       mov      rdi, 0x76079FF51990
       call     CORINFO_HELP_COUNTPROFILE32
       mov      eax, dword ptr [rbp-0x94]
       imul     eax, dword ptr [rbp-0x50]
       imul     eax, dword ptr [rbp+0x10]
       cdqe     
       mov      rcx, qword ptr [rbp-0x78]
       lea      rax, [rcx+4*rax]
       mov      qword ptr [rbp-0xA0], rax
       mov      eax, dword ptr [rbp+0x18]
       imul     eax, dword ptr [rbp+0x10]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0xA0]
       movsxd   rcx, dword ptr [rbp+0x20]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x6B8], rax
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x6F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x6F0]
       vmovups  ymmword ptr [rbp-0xD0], ymm0
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  ymm0, ymmword ptr [rax+0x20]
       vmovups  ymmword ptr [rbp-0x710], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x710]
       vmovups  ymmword ptr [rbp-0xF0], ymm0
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  ymm0, ymmword ptr [rax+0x08]
       vmovups  ymmword ptr [rbp-0x730], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x730]
       vmovups  ymmword ptr [rbp-0x110], ymm0
       mov      rax, qword ptr [rbp-0x6B8]
       vmovups  ymm0, ymmword ptr [rax+0x28]
       vmovups  ymmword ptr [rbp-0x750], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x750]
       vmovups  ymmword ptr [rbp-0x130], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0xD0]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0xF0], 32
       vmovups  ymmword ptr [rbp-0x150], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0xD0]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0xF0], 49
       vmovups  ymmword ptr [rbp-0x170], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x110]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x130], 32
       vmovups  ymmword ptr [rbp-0x190], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x110]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x130], 49
       vmovups  ymmword ptr [rbp-0x1B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x150]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x1D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x170]
 
G_M000_IG04:                ;; offset=0x0247
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x190]
       vmovups  ymmword ptr [rbp-0x1F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x190]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x170]
       vmovups  ymmword ptr [rbp-0x210], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x170]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x1B0]
       vmovups  ymmword ptr [rbp-0x230], ymm0
       mov      eax, dword ptr [rbp+0x18]
       add      eax, 2
       imul     eax, dword ptr [rbp+0x10]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0xA0]
       movsxd   rcx, dword ptr [rbp+0x20]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x758], rax
       mov      rax, qword ptr [rbp-0x758]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x790], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x790]
       vmovups  ymmword ptr [rbp-0x250], ymm0
       mov      rax, qword ptr [rbp-0x758]
       vmovups  ymm0, ymmword ptr [rax+0x20]
       vmovups  ymmword ptr [rbp-0x7B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x7B0]
       vmovups  ymmword ptr [rbp-0x270], ymm0
       mov      rax, qword ptr [rbp-0x758]
       vmovups  ymm0, ymmword ptr [rax+0x08]
       vmovups  ymmword ptr [rbp-0x7D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x7D0]
       vmovups  ymmword ptr [rbp-0x290], ymm0
       mov      rax, qword ptr [rbp-0x758]
       vmovups  ymm0, ymmword ptr [rax+0x28]
       vmovups  ymmword ptr [rbp-0x7F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x7F0]
       vmovups  ymmword ptr [rbp-0x2B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x250]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x270], 32
       vmovups  ymmword ptr [rbp-0x2D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x250]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x270], 49
       vmovups  ymmword ptr [rbp-0x2F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x290]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x2B0], 32
       vmovups  ymmword ptr [rbp-0x310], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x290]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x2B0], 49
       vmovups  ymmword ptr [rbp-0x330], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x2D0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x310]
       vmovups  ymmword ptr [rbp-0x350], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x2F0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x310]
 
G_M000_IG05:                ;; offset=0x03E4
       vmovups  ymmword ptr [rbp-0x370], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x310]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x2F0]
       vmovups  ymmword ptr [rbp-0x390], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x2F0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x330]
       vmovups  ymmword ptr [rbp-0x3B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x1D0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x350]
       mov      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x1F0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x370]
       mov      eax, dword ptr [rbp-0x4C]
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x210]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x390]
       mov      eax, dword ptr [rbp-0x4C]
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+2*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x230]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x3B0]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+2*rax]
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp+0x18]
       inc      eax
       imul     eax, dword ptr [rbp+0x10]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0xA0]
       movsxd   rcx, dword ptr [rbp+0x20]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x7F8], rax
       mov      rax, qword ptr [rbp-0x7F8]
       vmovups  ymm0, ymmword ptr [rax]
       vmovups  ymmword ptr [rbp-0x830], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x830]
       vmovups  ymmword ptr [rbp-0x3D0], ymm0
       mov      rax, qword ptr [rbp-0x7F8]
       vmovups  ymm0, ymmword ptr [rax+0x20]
       vmovups  ymmword ptr [rbp-0x850], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x850]
       vmovups  ymmword ptr [rbp-0x3F0], ymm0
 
G_M000_IG06:                ;; offset=0x0533
       mov      rax, qword ptr [rbp-0x7F8]
       vmovups  ymm0, ymmword ptr [rax+0x08]
       vmovups  ymmword ptr [rbp-0x870], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x870]
       vmovups  ymmword ptr [rbp-0x410], ymm0
       mov      rax, qword ptr [rbp-0x7F8]
       vmovups  ymm0, ymmword ptr [rax+0x28]
       vmovups  ymmword ptr [rbp-0x890], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x890]
       vmovups  ymmword ptr [rbp-0x430], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x3D0]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x3F0], 32
       vmovups  ymmword ptr [rbp-0x450], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x3D0]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x3F0], 49
       vmovups  ymmword ptr [rbp-0x470], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x410]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x430], 32
       vmovups  ymmword ptr [rbp-0x490], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x410]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x430], 49
       vmovups  ymmword ptr [rbp-0x4B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x450]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x490]
       vmovups  ymmword ptr [rbp-0x4D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x470]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x490]
       vmovups  ymmword ptr [rbp-0x4F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x490]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x470]
       vmovups  ymmword ptr [rbp-0x510], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x470]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x4B0]
       vmovups  ymmword ptr [rbp-0x530], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x4D0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x350]
       mov      eax, dword ptr [rbp-0x4C]
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+4*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x4F0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x370]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+4*rax]
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x510]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x390]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+2*rax]
 
G_M000_IG07:                ;; offset=0x06B9
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+2*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x530]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x3B0]
       imul     eax, dword ptr [rbp-0x4C], 7
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x350]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x4D0]
       mov      eax, dword ptr [rbp-0x4C]
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+8*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x370]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x4F0]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+8*rax]
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x390]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x510]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+4*rax]
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+2*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x3B0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x530]
       imul     eax, dword ptr [rbp-0x4C], 11
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp+0x18]
       add      eax, 3
       imul     eax, dword ptr [rbp+0x10]
       cdqe     
       shl      rax, 2
       add      rax, qword ptr [rbp-0xA0]
       movsxd   rcx, dword ptr [rbp+0x20]
       lea      rax, [rax+4*rcx]
       mov      qword ptr [rbp-0x898], rax
       mov      rax, qword ptr [rbp-0x898]
       vmovups  ymm0, ymmword ptr [rax]
 
G_M000_IG08:                ;; offset=0x07D2
       vmovups  ymmword ptr [rbp-0x8D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x8D0]
       vmovups  ymmword ptr [rbp-0x550], ymm0
       mov      rax, qword ptr [rbp-0x898]
       vmovups  ymm0, ymmword ptr [rax+0x20]
       vmovups  ymmword ptr [rbp-0x8F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x8F0]
       vmovups  ymmword ptr [rbp-0x570], ymm0
       mov      rax, qword ptr [rbp-0x898]
       vmovups  ymm0, ymmword ptr [rax+0x08]
       vmovups  ymmword ptr [rbp-0x910], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x910]
       vmovups  ymmword ptr [rbp-0x590], ymm0
       mov      rax, qword ptr [rbp-0x898]
       vmovups  ymm0, ymmword ptr [rax+0x28]
       vmovups  ymmword ptr [rbp-0x930], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x70]
       vpermps  ymm0, ymm0, ymmword ptr [rbp-0x930]
       vmovups  ymmword ptr [rbp-0x5B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x550]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x570], 32
       vmovups  ymmword ptr [rbp-0x5D0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x550]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x570], 49
       vmovups  ymmword ptr [rbp-0x5F0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x590]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x5B0], 32
       vmovups  ymmword ptr [rbp-0x610], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x590]
       vperm2f128 ymm0, ymm0, ymmword ptr [rbp-0x5B0], 49
       vmovups  ymmword ptr [rbp-0x630], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x5D0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x610]
       vmovups  ymmword ptr [rbp-0x650], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x5F0]
       vaddps   ymm0, ymm0, ymmword ptr [rbp-0x610]
       vmovups  ymmword ptr [rbp-0x670], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x610]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x5F0]
       vmovups  ymmword ptr [rbp-0x690], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x5F0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x630]
       vmovups  ymmword ptr [rbp-0x6B0], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x4D0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x650]
       mov      eax, dword ptr [rbp-0x4C]
       lea      eax, [rax+2*rax]
       mov      ecx, dword ptr [rbp-0x94]
       lea      eax, [rcx+4*rax]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x4F0]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x670]
 
G_M000_IG09:                ;; offset=0x0973
       imul     eax, dword ptr [rbp-0x4C], 13
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x510]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x690]
       imul     eax, dword ptr [rbp-0x4C], 14
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       vmovups  ymm0, ymmword ptr [rbp-0x530]
       vsubps   ymm0, ymm0, ymmword ptr [rbp-0x6B0]
       imul     eax, dword ptr [rbp-0x4C], 15
       add      eax, dword ptr [rbp-0x94]
       shl      eax, 3
       cdqe     
       mov      rcx, qword ptr [rbp-0x80]
       vmovups  ymmword ptr [rcx+4*rax], ymm0
       mov      eax, dword ptr [rbp-0x94]
       inc      eax
       mov      dword ptr [rbp-0x94], eax
 
G_M000_IG10:                ;; offset=0x09E9
       mov      eax, dword ptr [rbp-0x938]
       dec      eax
       mov      dword ptr [rbp-0x938], eax
       cmp      dword ptr [rbp-0x938], 0
       jg       SHORT G_M000_IG12
 
G_M000_IG11:                ;; offset=0x0A00
       lea      rdi, [rbp-0x938]
       mov      esi, 0x4D9
       call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG12:                ;; offset=0x0A11
       mov      eax, dword ptr [rbp-0x94]
       cmp      eax, dword ptr [rbp-0x4C]
       jl       G_M000_IG03
       mov      rdi, 0x76079FF51994
       call     CORINFO_HELP_COUNTPROFILE32
       xor      eax, eax
       mov      bword ptr [rbp-0x88], rax
       xor      eax, eax
       mov      bword ptr [rbp-0x90], rax
 
G_M000_IG13:                ;; offset=0x0A41
       vzeroupper 
       add      rsp, 0x950
       pop      rbp
       ret      
 
RWD00  	dq	0000000200000000h, 0000000600000004h, 0000000300000001h, 0000000700000005h

; Total bytes of code 2637

; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:TransformWinogradInputContiguous(System.ReadOnlySpan`1[float],System.Span`1[float],int,int,int,int,int) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 60704
; 2 inlinees with PGO data; 0 single block inlinees; 0 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       push     rbp
       push     r15
       push     r14
       push     r13
       push     rbx
       sub      rsp, 16
       lea      rbp, [rsp+0x30]
       xor      eax, eax
       mov      qword ptr [rbp-0x28], rax
       mov      qword ptr [rbp-0x30], rax
       mov      eax, dword ptr [rbp+0x10]
       mov      r10d, dword ptr [rbp+0x18]
       mov      r11d, dword ptr [rbp+0x20]
 
G_M000_IG02:                ;; offset=0x0026
       vmovups  ymm0, ymmword ptr [reloc @RWD00]
       xor      rbx, rbx
       test     esi, esi
       cmovne   rbx, rdi
       mov      bword ptr [rbp-0x28], rbx
       mov      rdi, rbx
       xor      rsi, rsi
       test     ecx, ecx
       cmovne   rsi, rdx
       mov      bword ptr [rbp-0x30], rsi
       mov      rcx, rsi
       xor      edx, edx
       cmp      edx, r8d
       jge      G_M000_IG06
       align    [0 bytes for IG03]
 
G_M000_IG03:                ;; offset=0x0057
       mov      esi, edx
       imul     esi, r9d
       imul     esi, eax
       movsxd   rsi, esi
       lea      rsi, [rdi+4*rsi]
       mov      ebx, r10d
       imul     ebx, eax
       movsxd   rbx, ebx
       shl      rbx, 2
       add      rbx, rsi
       movsxd   r15, r11d
       shl      r15, 2
       add      rbx, r15
       vpermps  ymm1, ymm0, ymmword ptr [rbx]
       vpermps  ymm2, ymm0, ymmword ptr [rbx+0x20]
       vpermps  ymm3, ymm0, ymmword ptr [rbx+0x08]
       vpermps  ymm4, ymm0, ymmword ptr [rbx+0x28]
       vperm2f128 ymm5, ymm1, ymm2, 32
       vperm2f128 ymm1, ymm1, ymm2, 49
       vperm2f128 ymm2, ymm3, ymm4, 32
       vperm2f128 ymm3, ymm3, ymm4, 49
       vsubps   ymm4, ymm5, ymm2
       vaddps   ymm5, ymm1, ymm2
       vsubps   ymm2, ymm2, ymm1
       vsubps   ymm1, ymm1, ymm3
       lea      ebx, [r10+0x02]
       imul     ebx, eax
       movsxd   rbx, ebx
       shl      rbx, 2
       add      rbx, rsi
       add      rbx, r15
       vpermps  ymm3, ymm0, ymmword ptr [rbx]
       vpermps  ymm6, ymm0, ymmword ptr [rbx+0x20]
       vpermps  ymm7, ymm0, ymmword ptr [rbx+0x08]
       vpermps  ymm8, ymm0, ymmword ptr [rbx+0x28]
       vperm2f128 ymm9, ymm3, ymm6, 32
       vperm2f128 ymm3, ymm3, ymm6, 49
       vperm2f128 ymm6, ymm7, ymm8, 32
       vperm2f128 ymm7, ymm7, ymm8, 49
       vsubps   ymm8, ymm9, ymm6
       vaddps   ymm9, ymm3, ymm6
       vsubps   ymm6, ymm6, ymm3
       vsubps   ymm3, ymm3, ymm7
       vsubps   ymm4, ymm4, ymm8
       lea      ebx, [8*rdx]
       movsxd   rbx, ebx
       vmovups  ymmword ptr [rcx+4*rbx], ymm4
       vsubps   ymm4, ymm5, ymm9
       lea      ebx, [r8+rdx]
       shl      ebx, 3
       movsxd   rbx, ebx
       vmovups  ymmword ptr [rcx+4*rbx], ymm4
       vsubps   ymm2, ymm2, ymm6
       lea      ebx, [rdx+2*r8]
       shl      ebx, 3
       movsxd   rbx, ebx
       vmovups  ymmword ptr [rcx+4*rbx], ymm2
       vsubps   ymm1, ymm1, ymm3
       lea      ebx, [r8+2*r8]
       lea      r14d, [rbx+rdx]
       shl      r14d, 3
       movsxd   r14, r14d
       vmovups  ymmword ptr [rcx+4*r14], ymm1
       lea      r14d, [r10+0x01]
 
G_M000_IG04:                ;; offset=0x016B
       imul     r14d, eax
       movsxd   r14, r14d
       shl      r14, 2
       add      r14, rsi
       add      r14, r15
       vpermps  ymm1, ymm0, ymmword ptr [r14]
       vpermps  ymm2, ymm0, ymmword ptr [r14+0x20]
       vpermps  ymm4, ymm0, ymmword ptr [r14+0x08]
       vpermps  ymm5, ymm0, ymmword ptr [r14+0x28]
       vperm2f128 ymm7, ymm1, ymm2, 32
       vperm2f128 ymm1, ymm1, ymm2, 49
       vperm2f128 ymm2, ymm4, ymm5, 32
       vperm2f128 ymm4, ymm4, ymm5, 49
       vsubps   ymm5, ymm7, ymm2
       vaddps   ymm7, ymm1, ymm2
       vsubps   ymm2, ymm2, ymm1
       vsubps   ymm1, ymm1, ymm4
       vaddps   ymm4, ymm5, ymm8
       lea      r14d, [rdx+4*r8]
       shl      r14d, 3
       movsxd   r14, r14d
       vmovups  ymmword ptr [rcx+4*r14], ymm4
       vaddps   ymm4, ymm7, ymm9
       lea      r14d, [r8+4*r8]
       lea      r13d, [r14+rdx]
       shl      r13d, 3
       movsxd   r13, r13d
       vmovups  ymmword ptr [rcx+4*r13], ymm4
       vaddps   ymm4, ymm2, ymm6
       lea      r13d, [rdx+2*rbx]
       shl      r13d, 3
       movsxd   r13, r13d
       vmovups  ymmword ptr [rcx+4*r13], ymm4
       vaddps   ymm4, ymm1, ymm3
       lea      r13d, [8*r8]
       sub      r13d, r8d
       add      r13d, edx
       shl      r13d, 3
       movsxd   r13, r13d
       vmovups  ymmword ptr [rcx+4*r13], ymm4
       vsubps   ymm4, ymm8, ymm5
       lea      r13d, [rdx+8*r8]
       shl      r13d, 3
       movsxd   r13, r13d
       vmovups  ymmword ptr [rcx+4*r13], ymm4
       vsubps   ymm4, ymm9, ymm7
       lea      r13d, [r8+8*r8]
       add      r13d, edx
       shl      r13d, 3
       movsxd   r13, r13d
       vmovups  ymmword ptr [rcx+4*r13], ymm4
       vsubps   ymm4, ymm6, ymm2
       lea      r14d, [rdx+2*r14]
       shl      r14d, 3
       movsxd   r14, r14d
       vmovups  ymmword ptr [rcx+4*r14], ymm4
       vsubps   ymm3, ymm3, ymm1
       imul     r14d, r8d, 11
       add      r14d, edx
       shl      r14d, 3
       movsxd   r14, r14d
       vmovups  ymmword ptr [rcx+4*r14], ymm3
       lea      r14d, [r10+0x03]
       imul     r14d, eax
       movsxd   r14, r14d
       shl      r14, 2
       add      rsi, r14
       add      rsi, r15
 
G_M000_IG05:                ;; offset=0x028E
       vpermps  ymm3, ymm0, ymmword ptr [rsi]
       vpermps  ymm4, ymm0, ymmword ptr [rsi+0x20]
       vpermps  ymm6, ymm0, ymmword ptr [rsi+0x08]
       vpermps  ymm8, ymm0, ymmword ptr [rsi+0x28]
       vperm2f128 ymm9, ymm3, ymm4, 32
       vperm2f128 ymm3, ymm3, ymm4, 49
       vperm2f128 ymm4, ymm6, ymm8, 32
       vperm2f128 ymm6, ymm6, ymm8, 49
       vsubps   ymm8, ymm9, ymm4
       vaddps   ymm9, ymm3, ymm4
       vsubps   ymm4, ymm4, ymm3
       vsubps   ymm3, ymm3, ymm6
       vsubps   ymm5, ymm5, ymm8
       lea      esi, [rdx+4*rbx]
       shl      esi, 3
       movsxd   rsi, esi
       vmovups  ymmword ptr [rcx+4*rsi], ymm5
       vsubps   ymm5, ymm7, ymm9
       imul     esi, r8d, 13
       add      esi, edx
       shl      esi, 3
       movsxd   rsi, esi
       vmovups  ymmword ptr [rcx+4*rsi], ymm5
       vsubps   ymm2, ymm2, ymm4
       imul     esi, r8d, 14
       add      esi, edx
       shl      esi, 3
       movsxd   rsi, esi
       vmovups  ymmword ptr [rcx+4*rsi], ymm2
       vsubps   ymm1, ymm1, ymm3
       mov      esi, r8d
       shl      esi, 4
       sub      esi, r8d
       add      esi, edx
       shl      esi, 3
       movsxd   rsi, esi
       vmovups  ymmword ptr [rcx+4*rsi], ymm1
       inc      edx
       cmp      edx, r8d
       jl       G_M000_IG03
 
G_M000_IG06:                ;; offset=0x0330
       xor      eax, eax
       mov      bword ptr [rbp-0x28], rax
 
G_M000_IG07:                ;; offset=0x0336
       mov      bword ptr [rbp-0x30], rax
 
G_M000_IG08:                ;; offset=0x033A
       vzeroupper 
       add      rsp, 16
       pop      rbx
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
RWD00  	dq	0000000200000000h, 0000000600000004h, 0000000300000001h, 0000000700000005h

; Total bytes of code 842

