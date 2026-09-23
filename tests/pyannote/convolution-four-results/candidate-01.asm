; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel512(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x1f7
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 1

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 64
       mov      qword ptr [rsp+0x7C8], r15
       mov      qword ptr [rsp+0x7C0], r14
       mov      qword ptr [rsp+0x7B8], r13
       mov      qword ptr [rsp+0x7B0], r12
       mov      qword ptr [rsp+0x7A8], rbx
       lea      rbp, [rsp+0x40]
       mov      rcx, qword ptr [rbp+0x760]
       mov      rdi, qword ptr [rbp+0x750]
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      eax, dword ptr [rbp+0x7A8]
       mov      r8d, dword ptr [rbp+0x7B8]
       mov      r12d, dword ptr [rbp+0x73C]
       mov      r13d, dword ptr [rbp+0x728]
       vmovups  zmm0, zmmword ptr [rbp+0x6E0]
       vmovups  zmm6, zmmword ptr [rbp+0x6A0]
       vmovups  zmm1, zmmword ptr [rbp+0x660]
       vmovups  zmm7, zmmword ptr [rbp+0x620]
       vmovups  zmm2, zmmword ptr [rbp+0x5E0]
       vmovups  zmm8, zmmword ptr [rbp+0x5A0]
       vmovups  zmm3, zmmword ptr [rbp+0x560]
       vmovups  zmm9, zmmword ptr [rbp+0x520]
       vmovups  zmm4, zmmword ptr [rbp+0x4E0]
       vmovups  zmm10, zmmword ptr [rbp+0x4A0]
       vmovups  zmm5, zmmword ptr [rbp+0x460]
       vmovups  zmm11, zmmword ptr [rbp+0x420]
       mov      rbx, qword ptr [rbp+0x418]
       mov      r15, qword ptr [rbp+0x410]
       mov      r10d, dword ptr [rbp+0x40C]
       mov      r14d, dword ptr [rbp+0x408]
       mov      r11d, dword ptr [rbp+0x404]
 
G_M000_IG02:                ;; offset=0x0106
       jmp      G_M000_IG10
 
G_M000_IG03:                ;; offset=0x010B
       inc      r10d
 
G_M000_IG04:                ;; offset=0x010E
       cmp      r10d, edx
       jge      G_M000_IG12
 
G_M000_IG05:                ;; offset=0x0117
       xor      r9d, r9d
       mov      r14d, r9d
       jmp      SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x011F
       inc      r14d
       cmp      r14d, 3
       jge      SHORT G_M000_IG03
 
G_M000_IG07:                ;; offset=0x0128
       xor      r9d, r9d
       mov      dword ptr [rbp+0x74C], edx
       mov      qword ptr [rbp+0x750], rdi
       mov      dword ptr [rbp+0x748], esi
       mov      dword ptr [rbp+0x7B8], r8d
       mov      r11d, r9d
 
G_M000_IG08:                ;; offset=0x0148
       vmovups  zmm12, zmmword ptr [rbx]
       vmovups  zmm13, zmmword ptr [r15]
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 15
       add      r9d, r10d
       sar      r9d, 4
       mov      r8d, dword ptr [rbp+0x740]
       imul     r9d, r8d
       mov      esi, dword ptr [rbp+0x72C]
       mov      edi, esi
       imul     edi, eax
       add      edi, r9d
       add      edi, r14d
       imul     edi, r12d
       mov      r9d, r13d
       imul     r9d, eax
       add      edi, r9d
       add      edi, r11d
       shl      edi, 4
       movsxd   rdi, edi
       shl      rdi, 2
       add      rdi, rcx
       mov      r9d, r10d
       sar      r9d, 31
       and      r9d, 15
       add      r9d, r10d
       and      r9d, -16
       mov      edx, r10d
       sub      edx, r9d
       movsxd   rdx, edx
       lea      rdx, [rdi+4*rdx]
       vbroadcastss zmm14, dword ptr [rdx]
       vfmadd231ps zmm0, zmm12, zmm14
       vfmadd231ps zmm6, zmm13, zmm14
       mov      edi, eax
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm1, zmm12, zmm14
       vfmadd231ps zmm7, zmm13, zmm14
       lea      edi, [rax+rax]
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm2, zmm12, zmm14
       vfmadd231ps zmm8, zmm13, zmm14
       imul     edi, eax, 48
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm3, zmm12, zmm14
       vfmadd231ps zmm9, zmm13, zmm14
       lea      edi, [4*rax]
       shl      edi, 4
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm4, zmm12, zmm14
       vfmadd231ps zmm10, zmm13, zmm14
       imul     edi, eax, 80
       movsxd   rdi, edi
       vbroadcastss zmm14, dword ptr [rdx+4*rdi]
       vfmadd231ps zmm5, zmm12, zmm14
       vfmadd231ps zmm11, zmm13, zmm14
       add      rbx, 64
       add      r15, 64
       inc      r11d
       mov      dword ptr [rbp+0x72C], esi
       mov      dword ptr [rbp+0x740], r8d
 
G_M000_IG09:                ;; offset=0x0272
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      rdi, qword ptr [rbp+0x750]
       mov      r8d, dword ptr [rbp+0x7B8]
 
G_M000_IG10:                ;; offset=0x028C
       cmp      r11d, 3
       jge      G_M000_IG06
 
G_M000_IG11:                ;; offset=0x0296
       mov      dword ptr [rbp+0x74C], edx
       mov      qword ptr [rbp+0x750], rdi
       mov      dword ptr [rbp+0x748], esi
       mov      dword ptr [rbp+0x7B8], r8d
       jmp      G_M000_IG08
 
G_M000_IG12:                ;; offset=0x02B5
       mov      r10d, dword ptr [rbp+0x730]
       mov      r11d, r10d
       sar      r11d, 31
       and      r11d, 15
       add      r11d, r10d
       sar      r11d, 4
       mov      r14d, dword ptr [rbp+0x738]
       mov      ebx, r11d
       imul     ebx, r14d
       mov      r15d, dword ptr [rbp+0x72C]
       mov      dword ptr [rbp+0x7B8], r8d
       mov      r9d, r15d
       imul     r9d, r8d
       add      ebx, r9d
       add      ebx, r13d
       mov      r8d, ebx
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm0
       mov      dword ptr [rbp+0x730], r10d
       lea      r8d, [r10+0x10]
       cmp      r8d, esi
       jge      SHORT G_M000_IG14
 
G_M000_IG13:                ;; offset=0x0318
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       add      r10d, r13d
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm6
 
G_M000_IG14:                ;; offset=0x0334
       lea      r10d, [rbx+0x01]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm1
       cmp      r8d, esi
       jge      SHORT G_M000_IG16
 
G_M000_IG15:                ;; offset=0x034B
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       lea      r10d, [r10+r13+0x01]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm7
 
G_M000_IG16:                ;; offset=0x0369
       lea      r10d, [rbx+0x02]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm2
       cmp      r8d, esi
       jge      SHORT G_M000_IG18
 
G_M000_IG17:                ;; offset=0x0380
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       lea      r10d, [r10+r13+0x02]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm8
 
G_M000_IG18:                ;; offset=0x039E
       lea      r10d, [rbx+0x03]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm3
       cmp      r8d, esi
       jge      SHORT G_M000_IG20
 
G_M000_IG19:                ;; offset=0x03B5
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       lea      r10d, [r10+r13+0x03]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm9
 
G_M000_IG20:                ;; offset=0x03D3
       lea      r10d, [rbx+0x04]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm4
       cmp      r8d, esi
       jge      SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x03EA
       lea      r10d, [r11+0x01]
       imul     r10d, r14d
       add      r10d, r9d
       lea      r10d, [r10+r13+0x04]
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rdi+4*r10], zmm10
 
G_M000_IG22:                ;; offset=0x0408
       add      ebx, 5
       shl      ebx, 4
       movsxd   r10, ebx
       vmovups  zmmword ptr [rdi+4*r10], zmm5
       mov      dword ptr [rbp+0x748], esi
       cmp      r8d, esi
       jge      SHORT G_M000_IG24
 
G_M000_IG23:                ;; offset=0x0423
       inc      r11d
       mov      dword ptr [rbp+0x738], r14d
       imul     r11d, r14d
       add      r9d, r11d
       lea      r8d, [r9+r13+0x05]
       shl      r8d, 4
       movsxd   r8, r8d
       mov      qword ptr [rbp+0x750], rdi
       vmovups  zmmword ptr [rdi+4*r8], zmm11
       mov      rdi, qword ptr [rbp+0x750]
       mov      r14d, dword ptr [rbp+0x738]
 
G_M000_IG24:                ;; offset=0x045C
       add      r13d, 6
 
G_M000_IG25:                ;; offset=0x0460
       lea      r8d, [r13+0x06]
       mov      r9d, dword ptr [rbp+0x7B8]
       cmp      r8d, r9d
       jg       G_M000_IG28
 
G_M000_IG26:                ;; offset=0x0474
       mov      r8d, r15d
       imul     r8d, r9d
       lea      r8d, [r8+r13+0x06]
       cmp      r8d, dword ptr [rbp+0x734]
       jg       G_M000_IG28
 
G_M000_IG27:                ;; offset=0x048D
       mov      dword ptr [rbp+0x7B8], r9d
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm7, ymm7, ymm7
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm8, ymm8, ymm8
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm9, ymm9, ymm9
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm10, ymm10, ymm10
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm11, ymm11, ymm11
       mov      ebx, dword ptr [rbp+0x730]
       mov      r8d, ebx
       imul     r8d, edx
       lea      r8d, [r8+8*r8]
       movsxd   r8, r8d
       mov      r10, qword ptr [rbp+0x758]
       lea      r8, [r10+4*r8]
       mov      dword ptr [rbp+0x74C], edx
       mov      r10d, edx
       shl      r10d, 4
       lea      r10d, [r10+8*r10]
       movsxd   r10, r10d
       lea      r10, [r8+4*r10]
       xor      r9d, r9d
       mov      dword ptr [rbp+0x72C], r15d
       mov      dword ptr [rbp+0x738], r14d
       mov      dword ptr [rbp+0x730], ebx
       mov      rbx, r8
       mov      r15, r10
       mov      r10d, r9d
       mov      edx, dword ptr [rbp+0x74C]
       mov      esi, dword ptr [rbp+0x748]
       mov      r8d, dword ptr [rbp+0x7B8]
       jmp      G_M000_IG04
 
G_M000_IG28:                ;; offset=0x0537
       mov      r8d, r15d
       imul     r8d, eax
       mov      dword ptr [rbp-0x30], r8d
       jmp      G_M000_IG35
       align    [0 bytes for IG29]
 
G_M000_IG29:                ;; offset=0x0547
       vmulps   zmm7, zmm1, zmm7
       vaddps   zmm0, zmm7, zmm0
       vmulps   zmm1, zmm1, zmm2
       vaddps   zmm6, zmm1, zmm6
       jmp      G_M000_IG43
 
G_M000_IG30:                ;; offset=0x0564
       vmulps   zmm8, zmm1, zmm7
       vaddps   zmm0, zmm8, zmm0
       vmulps   zmm3, zmm1, zmm2
       vaddps   zmm6, zmm3, zmm6
       jmp      G_M000_IG45
 
G_M000_IG31:                ;; offset=0x0581
       mov      r11d, dword ptr [rbp+0x740]
 
G_M000_IG32:                ;; offset=0x0588
       mov      r10d, dword ptr [rbp+0x730]
       mov      esi, r10d
       sar      esi, 31
       and      esi, 15
       add      esi, r10d
       sar      esi, 4
       mov      r14d, dword ptr [rbp+0x738]
       mov      r8d, esi
       imul     r8d, r14d
       mov      r9d, dword ptr [rbp+0x7B8]
       mov      ebx, r15d
       imul     ebx, r9d
       add      r8d, ebx
       add      r8d, r13d
       shl      r8d, 4
       movsxd   r8, r8d
       vmovups  zmmword ptr [rdi+4*r8], zmm0
       lea      r8d, [r10+0x10]
       mov      ebx, dword ptr [rbp+0x748]
       cmp      r8d, ebx
       jge      SHORT G_M000_IG34
 
G_M000_IG33:                ;; offset=0x05DD
       inc      esi
       imul     esi, r14d
       mov      dword ptr [rbp+0x7B8], r9d
       mov      r8d, r15d
       imul     r8d, r9d
       add      esi, r8d
       add      esi, r13d
       shl      esi, 4
       movsxd   rsi, esi
       mov      qword ptr [rbp+0x750], rdi
       vmovups  zmmword ptr [rdi+4*rsi], zmm6
       mov      rdi, qword ptr [rbp+0x750]
       mov      r9d, dword ptr [rbp+0x7B8]
 
G_M000_IG34:                ;; offset=0x0619
       inc      r13d
       mov      dword ptr [rbp+0x740], r11d
       mov      dword ptr [rbp+0x748], ebx
       mov      dword ptr [rbp+0x730], r10d
 
G_M000_IG35:                ;; offset=0x0630
       cmp      r13d, r9d
       jge      G_M000_IG48
 
G_M000_IG36:                ;; offset=0x0639
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm6, ymm6, ymm6
       mov      ebx, edx
       imul     ebx, dword ptr [rbp+0x730]
       lea      ebx, [rbx+8*rbx]
       movsxd   rbx, ebx
       mov      r10, qword ptr [rbp+0x758]
       lea      rbx, [r10+4*rbx]
       mov      r8d, edx
       shl      r8d, 4
       lea      r8d, [r8+8*r8]
       movsxd   r8, r8d
       lea      r8, [rbx+4*r8]
       mov      dword ptr [rbp+0x7B8], r9d
       mov      r10d, r15d
       imul     r10d, r9d
       add      r10d, r13d
       cmp      r10d, dword ptr [rbp+0x734]
       setl     r10b
       movzx    r10, r10b
       mov      dword ptr [rbp+0x2CC], r10d
       xor      r9d, r9d
       mov      r11d, r13d
       imul     r11d, eax
       mov      dword ptr [rbp-0x34], r11d
       cmp      r9d, edx
       mov      dword ptr [rbp+0x738], r14d
       jl       SHORT G_M000_IG39
       jmp      G_M000_IG31
 
G_M000_IG37:                ;; offset=0x06B3
       mov      rdi, qword ptr [rbp+0x750]
       inc      r9d
       cmp      r9d, edx
       jge      G_M000_IG32
 
G_M000_IG38:                ;; offset=0x06C6
       mov      dword ptr [rbp+0x740], r11d
 
G_M000_IG39:                ;; offset=0x06CD
       xor      r14d, r14d
       mov      esi, r9d
       sar      esi, 31
       and      esi, 15
       add      esi, r9d
       sar      esi, 4
       mov      r11d, dword ptr [rbp+0x740]
       imul     esi, r11d
       add      esi, dword ptr [rbp-0x30]
       mov      dword ptr [rbp-0x2C], esi
       mov      qword ptr [rbp+0x750], rdi
       jmp      G_M000_IG46
 
G_M000_IG40:                ;; offset=0x06FC
       vmulps   zmm7, zmm1, zmm7
       vaddps   zmm0, zmm7, zmm0
       vmulps   zmm1, zmm1, zmm2
       vaddps   zmm6, zmm1, zmm6
 
G_M000_IG41:                ;; offset=0x0714
       add      rbx, 64
       add      r8, 64
       lea      edi, [rsi+0x01]
       shl      edi, 4
       add      edi, r9d
       movsxd   rdi, edi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [rbx]
       vmovups  zmm2, zmmword ptr [r8]
       test     r10d, r10d
       je       G_M000_IG29
 
G_M000_IG42:                ;; offset=0x0744
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
 
G_M000_IG43:                ;; offset=0x0750
       add      rbx, 64
       add      r8, 64
       add      esi, 2
       shl      esi, 4
       add      esi, r9d
       movsxd   rdi, esi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [rbx]
       vmovups  zmm2, zmmword ptr [r8]
       test     r10d, r10d
       je       G_M000_IG30
 
G_M000_IG44:                ;; offset=0x0780
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
 
G_M000_IG45:                ;; offset=0x078C
       add      rbx, 64
       add      r8, 64
       inc      r14d
       cmp      r14d, 3
       mov      esi, dword ptr [rbp-0x2C]
       mov      r9d, dword ptr [rbp+0x2C8]
       jge      G_M000_IG37
 
G_M000_IG46:                ;; offset=0x07AB
       add      esi, r14d
       imul     esi, r12d
       add      esi, dword ptr [rbp-0x34]
       mov      edi, esi
       shl      edi, 4
       mov      r10d, r9d
       sar      r10d, 31
       and      r10d, 15
       add      r10d, r9d
       and      r10d, -16
       mov      dword ptr [rbp+0x2C8], r9d
       sub      r9d, r10d
       add      edi, r9d
       movsxd   rdi, edi
       vbroadcastss zmm1, dword ptr [rcx+4*rdi]
       vmovups  zmm7, zmmword ptr [rbx]
       vmovups  zmm2, zmmword ptr [r8]
       mov      r10d, dword ptr [rbp+0x2CC]
       test     r10d, r10d
       je       G_M000_IG40
 
G_M000_IG47:                ;; offset=0x07FF
       vfmadd231ps zmm0, zmm7, zmm1
       vfmadd231ps zmm6, zmm2, zmm1
       jmp      G_M000_IG41
 
G_M000_IG48:                ;; offset=0x0810
       inc      r15d
       mov      r8d, dword ptr [rbp+0x7B0]
       cmp      r15d, r8d
       jge      SHORT G_M000_IG52
 
G_M000_IG49:                ;; offset=0x081F
       xor      r13d, r13d
       mov      dword ptr [rbp+0x7B8], r9d
       mov      dword ptr [rbp+0x7B0], r8d
       jmp      G_M000_IG25
 
G_M000_IG50:                ;; offset=0x0835
       xor      r13d, r13d
       mov      dword ptr [rbp+0x7B0], r8d
       test     r8d, r8d
       mov      dword ptr [rbp+0x730], r15d
       mov      r8d, dword ptr [rbp+0x7B0]
       jle      SHORT G_M000_IG52
 
G_M000_IG51:                ;; offset=0x0852
       mov      r15d, r13d
       jmp      SHORT G_M000_IG49
 
G_M000_IG52:                ;; offset=0x0857
       mov      r15d, dword ptr [rbp+0x730]
       add      r15d, 32
       cmp      r15d, dword ptr [rbp+0x748]
       jl       SHORT G_M000_IG50
 
G_M000_IG53:                ;; offset=0x086B
       vzeroupper 
       add      rsp, 0x7A8
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 2176

