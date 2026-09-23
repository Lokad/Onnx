; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:PlanWinograd(int,int,int,int,byref,byref,byref):bool (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Dynamic PGO
; rbp based frame
; fully interruptible
; with Dynamic PGO: fgCalledCount is 164

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

