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

