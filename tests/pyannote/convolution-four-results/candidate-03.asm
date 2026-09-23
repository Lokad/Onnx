; Assembly listing for method Lokad.Onnx.ConvBlockedSpatial:Kernel512Four(ptr,ptr,ptr,int,int,int,int,int,int,int) (Tier1-OSR)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1-OSR code
; OSR variant for entry point 0x1101
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 1

G_M000_IG01:                ;; offset=0x0000
       mov      rax, qword ptr [rbp]
       push     rax
       sub      rsp, 64
       mov      qword ptr [rsp+0x2288], r15
       mov      qword ptr [rsp+0x2280], r14
       mov      qword ptr [rsp+0x2278], r13
       mov      qword ptr [rsp+0x2270], r12
       mov      qword ptr [rsp+0x2268], rbx
       lea      rbp, [rsp+0x40]
       mov      rdx, qword ptr [rbp+0x2220]
       mov      rcx, qword ptr [rbp+0x2210]
       mov      eax, dword ptr [rbp+0x220C]
       mov      edi, dword ptr [rbp+0x2278]
       mov      r13d, dword ptr [rbp+0x21E8]
       vmovups  zmm0, zmmword ptr [rbp+0x2160]
       vmovups  zmm1, zmmword ptr [rbp+0x2120]
       vmovups  zmm2, zmmword ptr [rbp+0x20E0]
       vmovups  zmm3, zmmword ptr [rbp+0x20A0]
       vmovups  zmm4, zmmword ptr [rbp+0x2060]
       vmovups  zmm5, zmmword ptr [rbp+0x2020]
       vmovups  zmm6, zmmword ptr [rbp+0x1FE0]
       vmovups  zmm7, zmmword ptr [rbp+0x1FA0]
       vmovups  zmm8, zmmword ptr [rbp+0x1F60]
       vmovups  zmm9, zmmword ptr [rbp+0x1F20]
       vmovups  zmm10, zmmword ptr [rbp+0x1EE0]
       vmovups  zmm11, zmmword ptr [rbp+0x1EA0]
       vmovups  zmm12, zmmword ptr [rbp+0x1E60]
       vmovups  zmm13, zmmword ptr [rbp+0x1E20]
       vmovups  zmm14, zmmword ptr [rbp+0x1DE0]
       vmovups  zmm15, zmmword ptr [rbp+0x1DA0]
       vmovups  zmm16, zmmword ptr [rbp+0x1D60]
       vmovups  zmm17, zmmword ptr [rbp+0x1D20]
       vmovups  zmm18, zmmword ptr [rbp+0x1CE0]
       vmovups  zmm19, zmmword ptr [rbp+0x1CA0]
       vmovups  zmm20, zmmword ptr [rbp+0x1C60]
       vmovups  zmm21, zmmword ptr [rbp+0x1C20]
       vmovups  zmm22, zmmword ptr [rbp+0x1BE0]
       vmovups  zmm23, zmmword ptr [rbp+0x1BA0]
       mov      r10, qword ptr [rbp+0x1B98]
       mov      r11, qword ptr [rbp+0x1B90]
       mov      rbx, qword ptr [rbp+0x1B88]
       mov      r15, qword ptr [rbp+0x1B80]
       mov      r12, qword ptr [rbp+0x1B78]
       mov      r14d, dword ptr [rbp+0x1B74]
 
G_M000_IG02:                ;; offset=0x0171
       mov      r9d, eax
       shl      r9d, 4
       lea      r9d, [r9+8*r9]
       movsxd   r9, r9d
       shl      r9, 2
       mov      qword ptr [rbp-0x38], r9
       jmp      G_M000_IG17
       align    [0 bytes for IG24]
 
G_M000_IG03:                ;; offset=0x018C
       mov      eax, dword ptr [rbp+0x220C]
       mov      rdx, qword ptr [rbp+0x2220]
       mov      edi, dword ptr [rbp+0x2278]
 
G_M000_IG04:                ;; offset=0x019F
       mov      r10d, dword ptr [rbp+0x21E4]
       mov      r11d, r10d
       sar      r11d, 31
       and      r11d, 15
       mov      dword ptr [rbp+0x21E4], r10d
       add      r11d, r10d
       sar      r11d, 4
       mov      r14d, dword ptr [rbp+0x21F8]
       mov      ebx, r11d
       imul     ebx, r14d
       mov      r15d, dword ptr [rbp+0x21E0]
       mov      r12d, r15d
       imul     r12d, edi
       add      ebx, r12d
       mov      r8d, dword ptr [rbp+0x21DC]
       add      ebx, r8d
       mov      esi, ebx
       shl      esi, 4
       movsxd   rsi, esi
       vmovups  zmmword ptr [rcx+4*rsi], zmm0
       lea      esi, [r11+0x01]
       imul     esi, r14d
       add      esi, r12d
       add      esi, r8d
       mov      r10d, esi
       shl      r10d, 4
       movsxd   r10, r10d
       vmovups  zmmword ptr [rcx+4*r10], zmm1
       lea      r10d, [r11+0x02]
       imul     r10d, r14d
       add      r10d, r12d
       add      r10d, r8d
       mov      r9d, r10d
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm2
       add      r11d, 3
       mov      dword ptr [rbp+0x21F8], r14d
       imul     r11d, r14d
       add      r11d, r12d
       add      r11d, r8d
       mov      r9d, r11d
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm3
       lea      r9d, [rbx+0x01]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm4
       lea      r9d, [rsi+0x01]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm5
       lea      r9d, [r10+0x01]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm6
       lea      r9d, [r11+0x01]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm7
       lea      r9d, [rbx+0x02]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm8
       lea      r9d, [rsi+0x02]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm9
       lea      r9d, [r10+0x02]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm10
       lea      r9d, [r11+0x02]
 
G_M000_IG05:                ;; offset=0x02DD
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm11
       lea      r9d, [rbx+0x03]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm12
       lea      r9d, [rsi+0x03]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm13
       lea      r9d, [r10+0x03]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm14
       lea      r9d, [r11+0x03]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm15
       lea      r9d, [rbx+0x04]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm16
       lea      r9d, [rsi+0x04]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm17
       lea      r9d, [r10+0x04]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm18
       lea      r9d, [r11+0x04]
       shl      r9d, 4
       movsxd   r9, r9d
       vmovups  zmmword ptr [rcx+4*r9], zmm19
       add      ebx, 5
       shl      ebx, 4
       movsxd   r9, ebx
       vmovups  zmmword ptr [rcx+4*r9], zmm20
       add      esi, 5
       shl      esi, 4
       movsxd   rsi, esi
       vmovups  zmmword ptr [rcx+4*rsi], zmm21
       add      r10d, 5
       shl      r10d, 4
       movsxd   rsi, r10d
       vmovups  zmmword ptr [rcx+4*rsi], zmm22
       add      r11d, 5
       shl      r11d, 4
       movsxd   rsi, r11d
       vmovups  zmmword ptr [rcx+4*rsi], zmm23
       add      r8d, 6
 
G_M000_IG06:                ;; offset=0x03C3
       lea      esi, [r8+0x06]
       cmp      esi, edi
       jg       G_M000_IG19
 
G_M000_IG07:                ;; offset=0x03CF
       mov      esi, r15d
       imul     esi, edi
       lea      esi, [rsi+r8+0x06]
       cmp      esi, dword ptr [rbp+0x21F4]
       jg       G_M000_IG19
 
G_M000_IG08:                ;; offset=0x03E6
       mov      dword ptr [rbp+0x2278], edi
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       vxorps   ymm4, ymm4, ymm4
       vxorps   ymm5, ymm5, ymm5
       vxorps   ymm6, ymm6, ymm6
       vxorps   ymm7, ymm7, ymm7
       vxorps   ymm8, ymm8, ymm8
       vxorps   ymm9, ymm9, ymm9
       vxorps   ymm10, ymm10, ymm10
       vxorps   ymm11, ymm11, ymm11
       vxorps   ymm12, ymm12, ymm12
       vxorps   ymm13, ymm13, ymm13
       vxorps   ymm14, ymm14, ymm14
       vxorps   ymm15, ymm15, ymm15
       vxorps   ymm16, ymm16, ymm16
       vxorps   ymm17, ymm17, ymm17
       vxorps   ymm18, ymm18, ymm18
       vxorps   ymm19, ymm19, ymm19
       vxorps   ymm20, ymm20, ymm20
       vxorps   ymm21, ymm21, ymm21
       vxorps   ymm22, ymm22, ymm22
       vxorps   ymm23, ymm23, ymm23
       mov      r11d, dword ptr [rbp+0x21E4]
       mov      ebx, r11d
       imul     ebx, eax
       lea      ebx, [rbx+8*rbx]
       movsxd   rbx, ebx
       mov      r12, qword ptr [rbp+0x2218]
       lea      rbx, [r12+4*rbx]
       mov      r10, qword ptr [rbp-0x38]
       lea      r9, [rbx+r10]
       lea      r12, [r9+r10]
       lea      rsi, [r12+r10]
       mov      r10d, r15d
       imul     r10d, dword ptr [rbp+0x2268]
       mov      edi, dword ptr [rbp+0x21FC]
       imul     r10d, edi
       mov      r14d, r8d
       imul     r14d, dword ptr [rbp+0x2268]
       add      r10d, r14d
       shl      r10d, 4
       movsxd   r10, r10d
       mov      qword ptr [rbp+0x2220], rdx
       lea      r10, [rdx+4*r10]
       xor      r14d, r14d
       mov      dword ptr [rbp+0x220C], eax
       cmp      r14d, eax
       mov      dword ptr [rbp+0x21DC], r8d
       mov      dword ptr [rbp+0x21FC], edi
       mov      dword ptr [rbp+0x21E0], r15d
       mov      dword ptr [rbp+0x21E4], r11d
       jge      G_M000_IG03
 
G_M000_IG09:                ;; offset=0x04F4
       mov      qword ptr [rbp+0x2210], rcx
       mov      r11, r9
       mov      r15, rsi
       mov      rax, rbx
       mov      rbx, r12
       mov      r12, r10
       mov      r10, rax
 
G_M000_IG10:                ;; offset=0x050D
       mov      r8d, r14d
       sar      r8d, 4
       imul     r8d, dword ptr [rbp+0x21EC]
       movsxd   r8, r8d
       shl      r8, 2
       mov      qword ptr [rbp+0x1B78], r12
       add      r8, r12
       mov      r9d, r14d
       and      r9d, 15
       movsxd   r9, r9d
       lea      r8, [r8+4*r9]
       movsxd   rdi, dword ptr [rbp+0x21F0]
       shl      rdi, 2
       lea      rdx, [r8+rdi]
       add      rdi, rdx
       vmovups  zmm24, zmmword ptr [r10]
       vmovups  zmm25, zmmword ptr [r11]
       vmovups  zmm26, zmmword ptr [rbx]
       vmovups  zmm27, zmmword ptr [r15]
       vbroadcastss zmm28, dword ptr [r8]
       vfmadd231ps zmm0, zmm24, zmm28
       vfmadd231ps zmm1, zmm25, zmm28
       vfmadd231ps zmm2, zmm26, zmm28
       vfmadd231ps zmm3, zmm27, zmm28
       movsxd   rcx, r13d
       vbroadcastss zmm28, dword ptr [r8+4*rcx]
       vfmadd231ps zmm4, zmm24, zmm28
       vfmadd231ps zmm5, zmm25, zmm28
       vfmadd231ps zmm6, zmm26, zmm28
       vfmadd231ps zmm7, zmm27, zmm28
       lea      r9d, [r13+r13]
       movsxd   r9, r9d
       vbroadcastss zmm28, dword ptr [r8+4*r9]
       vfmadd231ps zmm8, zmm24, zmm28
       vfmadd231ps zmm9, zmm25, zmm28
       vfmadd231ps zmm10, zmm26, zmm28
       vfmadd231ps zmm11, zmm27, zmm28
       lea      esi, [r13+2*r13]
       movsxd   rsi, esi
       vbroadcastss zmm28, dword ptr [r8+4*rsi]
       vfmadd231ps zmm12, zmm24, zmm28
       vfmadd231ps zmm13, zmm25, zmm28
       vfmadd231ps zmm14, zmm26, zmm28
       vfmadd231ps zmm15, zmm27, zmm28
       lea      r12d, [4*r13]
       movsxd   r12, r12d
       vbroadcastss zmm28, dword ptr [r8+4*r12]
       vfmadd231ps zmm16, zmm24, zmm28
       vfmadd231ps zmm17, zmm25, zmm28
       vfmadd231ps zmm18, zmm26, zmm28
       vfmadd231ps zmm19, zmm27, zmm28
       mov      dword ptr [rbp+0x21E8], r13d
       lea      eax, [r13+4*r13]
       cdqe     
       vbroadcastss zmm28, dword ptr [r8+4*rax]
       vfmadd231ps zmm20, zmm24, zmm28
       vfmadd231ps zmm21, zmm25, zmm28
       vfmadd231ps zmm22, zmm26, zmm28
       vfmadd231ps zmm23, zmm27, zmm28
       add      r10, 64
       add      r11, 64
       add      rbx, 64
 
G_M000_IG11:                ;; offset=0x0656
       add      r15, 64
       vmovups  zmm24, zmmword ptr [r10]
       vmovups  zmm25, zmmword ptr [r11]
       vmovups  zmm26, zmmword ptr [rbx]
       vmovups  zmm27, zmmword ptr [r15]
       lea      r13, [r8+0x40]
       vbroadcastss zmm28, dword ptr [r13]
       vfmadd231ps zmm0, zmm24, zmm28
       vfmadd231ps zmm1, zmm25, zmm28
       vfmadd231ps zmm2, zmm26, zmm28
       vfmadd231ps zmm3, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r13+4*rcx]
       vfmadd231ps zmm4, zmm24, zmm28
       vfmadd231ps zmm5, zmm25, zmm28
       vfmadd231ps zmm6, zmm26, zmm28
       vfmadd231ps zmm7, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r13+4*r9]
       vfmadd231ps zmm8, zmm24, zmm28
       vfmadd231ps zmm9, zmm25, zmm28
       vfmadd231ps zmm10, zmm26, zmm28
       vfmadd231ps zmm11, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r13+4*rsi]
       vfmadd231ps zmm12, zmm24, zmm28
       vfmadd231ps zmm13, zmm25, zmm28
       vfmadd231ps zmm14, zmm26, zmm28
       vfmadd231ps zmm15, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r13+4*r12]
       vfmadd231ps zmm16, zmm24, zmm28
       vfmadd231ps zmm17, zmm25, zmm28
       vfmadd231ps zmm18, zmm26, zmm28
       vfmadd231ps zmm19, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r13+4*rax]
       vfmadd231ps zmm20, zmm24, zmm28
       vfmadd231ps zmm21, zmm25, zmm28
       vfmadd231ps zmm22, zmm26, zmm28
       vfmadd231ps zmm23, zmm27, zmm28
       add      r10, 64
       add      r11, 64
       add      rbx, 64
       add      r15, 64
       vmovups  zmm24, zmmword ptr [r10]
       vmovups  zmm25, zmmword ptr [r11]
       vmovups  zmm26, zmmword ptr [rbx]
       vmovups  zmm27, zmmword ptr [r15]
       add      r8, 128
       vbroadcastss zmm28, dword ptr [r8]
       vfmadd231ps zmm0, zmm24, zmm28
       vfmadd231ps zmm1, zmm25, zmm28
       vfmadd231ps zmm2, zmm26, zmm28
       vfmadd231ps zmm3, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r8+4*rcx]
       vfmadd231ps zmm4, zmm24, zmm28
       vfmadd231ps zmm5, zmm25, zmm28
       vfmadd231ps zmm6, zmm26, zmm28
       vfmadd231ps zmm7, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r8+4*r9]
       vfmadd231ps zmm8, zmm24, zmm28
 
G_M000_IG12:                ;; offset=0x07AE
       vfmadd231ps zmm9, zmm25, zmm28
       vfmadd231ps zmm10, zmm26, zmm28
       vfmadd231ps zmm11, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r8+4*rsi]
       vfmadd231ps zmm12, zmm24, zmm28
       vfmadd231ps zmm13, zmm25, zmm28
       vfmadd231ps zmm14, zmm26, zmm28
       vfmadd231ps zmm15, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r8+4*r12]
       vfmadd231ps zmm16, zmm24, zmm28
       vfmadd231ps zmm17, zmm25, zmm28
       vfmadd231ps zmm18, zmm26, zmm28
       vfmadd231ps zmm19, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r8+4*rax]
       vfmadd231ps zmm20, zmm24, zmm28
       vfmadd231ps zmm21, zmm25, zmm28
       vfmadd231ps zmm22, zmm26, zmm28
       vfmadd231ps zmm23, zmm27, zmm28
       add      r10, 64
       add      r11, 64
       add      rbx, 64
       add      r15, 64
       vmovups  zmm24, zmmword ptr [r10]
       vmovups  zmm25, zmmword ptr [r11]
       vmovups  zmm26, zmmword ptr [rbx]
       vmovups  zmm27, zmmword ptr [r15]
       vbroadcastss zmm28, dword ptr [rdx]
       vfmadd231ps zmm0, zmm24, zmm28
       vfmadd231ps zmm1, zmm25, zmm28
       vfmadd231ps zmm2, zmm26, zmm28
       vfmadd231ps zmm3, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*rcx]
       vfmadd231ps zmm4, zmm24, zmm28
       vfmadd231ps zmm5, zmm25, zmm28
       vfmadd231ps zmm6, zmm26, zmm28
       vfmadd231ps zmm7, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*r9]
       vfmadd231ps zmm8, zmm24, zmm28
       vfmadd231ps zmm9, zmm25, zmm28
       vfmadd231ps zmm10, zmm26, zmm28
       vfmadd231ps zmm11, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*rsi]
       vfmadd231ps zmm12, zmm24, zmm28
       vfmadd231ps zmm13, zmm25, zmm28
       vfmadd231ps zmm14, zmm26, zmm28
       vfmadd231ps zmm15, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*r12]
       vfmadd231ps zmm16, zmm24, zmm28
       vfmadd231ps zmm17, zmm25, zmm28
       vfmadd231ps zmm18, zmm26, zmm28
       vfmadd231ps zmm19, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*rax]
       vfmadd231ps zmm20, zmm24, zmm28
       vfmadd231ps zmm21, zmm25, zmm28
       vfmadd231ps zmm22, zmm26, zmm28
       vfmadd231ps zmm23, zmm27, zmm28
       add      r10, 64
 
G_M000_IG13:                ;; offset=0x0902
       add      r11, 64
       add      rbx, 64
       add      r15, 64
       vmovups  zmm24, zmmword ptr [r10]
       vmovups  zmm25, zmmword ptr [r11]
       vmovups  zmm26, zmmword ptr [rbx]
       vmovups  zmm27, zmmword ptr [r15]
       lea      r8, [rdx+0x40]
       vbroadcastss zmm28, dword ptr [r8]
       vfmadd231ps zmm0, zmm24, zmm28
       vfmadd231ps zmm1, zmm25, zmm28
       vfmadd231ps zmm2, zmm26, zmm28
       vfmadd231ps zmm3, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r8+4*rcx]
       vfmadd231ps zmm4, zmm24, zmm28
       vfmadd231ps zmm5, zmm25, zmm28
       vfmadd231ps zmm6, zmm26, zmm28
       vfmadd231ps zmm7, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r8+4*r9]
       vfmadd231ps zmm8, zmm24, zmm28
       vfmadd231ps zmm9, zmm25, zmm28
       vfmadd231ps zmm10, zmm26, zmm28
       vfmadd231ps zmm11, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r8+4*rsi]
       vfmadd231ps zmm12, zmm24, zmm28
       vfmadd231ps zmm13, zmm25, zmm28
       vfmadd231ps zmm14, zmm26, zmm28
       vfmadd231ps zmm15, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r8+4*r12]
       vfmadd231ps zmm16, zmm24, zmm28
       vfmadd231ps zmm17, zmm25, zmm28
       vfmadd231ps zmm18, zmm26, zmm28
       vfmadd231ps zmm19, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [r8+4*rax]
       vfmadd231ps zmm20, zmm24, zmm28
       vfmadd231ps zmm21, zmm25, zmm28
       vfmadd231ps zmm22, zmm26, zmm28
       vfmadd231ps zmm23, zmm27, zmm28
       add      r10, 64
       add      r11, 64
       add      rbx, 64
       add      r15, 64
       vmovups  zmm24, zmmword ptr [r10]
       vmovups  zmm25, zmmword ptr [r11]
       vmovups  zmm26, zmmword ptr [rbx]
       vmovups  zmm27, zmmword ptr [r15]
       add      rdx, 128
       vbroadcastss zmm28, dword ptr [rdx]
       vfmadd231ps zmm0, zmm24, zmm28
       vfmadd231ps zmm1, zmm25, zmm28
       vfmadd231ps zmm2, zmm26, zmm28
       vfmadd231ps zmm3, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*rcx]
       vfmadd231ps zmm4, zmm24, zmm28
       vfmadd231ps zmm5, zmm25, zmm28
       vfmadd231ps zmm6, zmm26, zmm28
 
G_M000_IG14:                ;; offset=0x0A49
       vfmadd231ps zmm7, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*r9]
       vfmadd231ps zmm8, zmm24, zmm28
       vfmadd231ps zmm9, zmm25, zmm28
       vfmadd231ps zmm10, zmm26, zmm28
       vfmadd231ps zmm11, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*rsi]
       vfmadd231ps zmm12, zmm24, zmm28
       vfmadd231ps zmm13, zmm25, zmm28
       vfmadd231ps zmm14, zmm26, zmm28
       vfmadd231ps zmm15, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*r12]
       vfmadd231ps zmm16, zmm24, zmm28
       vfmadd231ps zmm17, zmm25, zmm28
       vfmadd231ps zmm18, zmm26, zmm28
       vfmadd231ps zmm19, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*rax]
       vfmadd231ps zmm20, zmm24, zmm28
       vfmadd231ps zmm21, zmm25, zmm28
       vfmadd231ps zmm22, zmm26, zmm28
       vfmadd231ps zmm23, zmm27, zmm28
       add      r10, 64
       add      r11, 64
       add      rbx, 64
       add      r15, 64
       vmovups  zmm24, zmmword ptr [r10]
       vmovups  zmm25, zmmword ptr [r11]
       vmovups  zmm26, zmmword ptr [rbx]
       vmovups  zmm27, zmmword ptr [r15]
       vbroadcastss zmm28, dword ptr [rdi]
       vfmadd231ps zmm0, zmm24, zmm28
       vfmadd231ps zmm1, zmm25, zmm28
       vfmadd231ps zmm2, zmm26, zmm28
       vfmadd231ps zmm3, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdi+4*rcx]
       vfmadd231ps zmm4, zmm24, zmm28
       vfmadd231ps zmm5, zmm25, zmm28
       vfmadd231ps zmm6, zmm26, zmm28
       vfmadd231ps zmm7, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdi+4*r9]
       vfmadd231ps zmm8, zmm24, zmm28
       vfmadd231ps zmm9, zmm25, zmm28
       vfmadd231ps zmm10, zmm26, zmm28
       vfmadd231ps zmm11, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdi+4*rsi]
       vfmadd231ps zmm12, zmm24, zmm28
       vfmadd231ps zmm13, zmm25, zmm28
       vfmadd231ps zmm14, zmm26, zmm28
       vfmadd231ps zmm15, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdi+4*r12]
       vfmadd231ps zmm16, zmm24, zmm28
       vfmadd231ps zmm17, zmm25, zmm28
       vfmadd231ps zmm18, zmm26, zmm28
       vfmadd231ps zmm19, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdi+4*rax]
       vfmadd231ps zmm20, zmm24, zmm28
       vfmadd231ps zmm21, zmm25, zmm28
 
G_M000_IG15:                ;; offset=0x0BA0
       vfmadd231ps zmm22, zmm26, zmm28
       vfmadd231ps zmm23, zmm27, zmm28
       add      r10, 64
       add      r11, 64
       add      rbx, 64
       add      r15, 64
       vmovups  zmm24, zmmword ptr [r10]
       vmovups  zmm25, zmmword ptr [r11]
       vmovups  zmm26, zmmword ptr [rbx]
       vmovups  zmm27, zmmword ptr [r15]
       lea      rdx, [rdi+0x40]
       vbroadcastss zmm28, dword ptr [rdx]
       vfmadd231ps zmm0, zmm24, zmm28
       vfmadd231ps zmm1, zmm25, zmm28
       vfmadd231ps zmm2, zmm26, zmm28
       vfmadd231ps zmm3, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*rcx]
       vfmadd231ps zmm4, zmm24, zmm28
       vfmadd231ps zmm5, zmm25, zmm28
       vfmadd231ps zmm6, zmm26, zmm28
       vfmadd231ps zmm7, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*r9]
       vfmadd231ps zmm8, zmm24, zmm28
       vfmadd231ps zmm9, zmm25, zmm28
       vfmadd231ps zmm10, zmm26, zmm28
       vfmadd231ps zmm11, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*rsi]
       vfmadd231ps zmm12, zmm24, zmm28
       vfmadd231ps zmm13, zmm25, zmm28
       vfmadd231ps zmm14, zmm26, zmm28
       vfmadd231ps zmm15, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*r12]
       vfmadd231ps zmm16, zmm24, zmm28
       vfmadd231ps zmm17, zmm25, zmm28
       vfmadd231ps zmm18, zmm26, zmm28
       vfmadd231ps zmm19, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdx+4*rax]
       vfmadd231ps zmm20, zmm24, zmm28
       vfmadd231ps zmm21, zmm25, zmm28
       vfmadd231ps zmm22, zmm26, zmm28
       vfmadd231ps zmm23, zmm27, zmm28
       add      r10, 64
       add      r11, 64
       add      rbx, 64
       add      r15, 64
       vmovups  zmm24, zmmword ptr [r10]
       vmovups  zmm25, zmmword ptr [r11]
       vmovups  zmm26, zmmword ptr [rbx]
       vmovups  zmm27, zmmword ptr [r15]
       add      rdi, 128
       vbroadcastss zmm28, dword ptr [rdi]
       vfmadd231ps zmm0, zmm24, zmm28
       vfmadd231ps zmm1, zmm25, zmm28
       vfmadd231ps zmm2, zmm26, zmm28
       vfmadd231ps zmm3, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdi+4*rcx]
 
G_M000_IG16:                ;; offset=0x0CE5
       vfmadd231ps zmm4, zmm24, zmm28
       vfmadd231ps zmm5, zmm25, zmm28
       vfmadd231ps zmm6, zmm26, zmm28
       vfmadd231ps zmm7, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdi+4*r9]
       vfmadd231ps zmm8, zmm24, zmm28
       vfmadd231ps zmm9, zmm25, zmm28
       vfmadd231ps zmm10, zmm26, zmm28
       vfmadd231ps zmm11, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdi+4*rsi]
       vfmadd231ps zmm12, zmm24, zmm28
       vfmadd231ps zmm13, zmm25, zmm28
       vfmadd231ps zmm14, zmm26, zmm28
       vfmadd231ps zmm15, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdi+4*r12]
       vfmadd231ps zmm16, zmm24, zmm28
       vfmadd231ps zmm17, zmm25, zmm28
       vfmadd231ps zmm18, zmm26, zmm28
       vfmadd231ps zmm19, zmm27, zmm28
       vbroadcastss zmm28, dword ptr [rdi+4*rax]
       vfmadd231ps zmm20, zmm24, zmm28
       vfmadd231ps zmm21, zmm25, zmm28
       vfmadd231ps zmm22, zmm26, zmm28
       vfmadd231ps zmm23, zmm27, zmm28
       add      r10, 64
       add      r11, 64
       add      rbx, 64
       add      r15, 64
       inc      r14d
       mov      eax, dword ptr [rbp+0x220C]
       mov      rcx, qword ptr [rbp+0x2210]
       mov      rdx, qword ptr [rbp+0x2220]
       mov      edi, dword ptr [rbp+0x2278]
       mov      r12, qword ptr [rbp+0x1B78]
       mov      r13d, dword ptr [rbp+0x21E8]
 
G_M000_IG17:                ;; offset=0x0DB4
       cmp      r14d, eax
       jge      G_M000_IG04
 
G_M000_IG18:                ;; offset=0x0DBD
       mov      dword ptr [rbp+0x220C], eax
       mov      qword ptr [rbp+0x2210], rcx
       mov      qword ptr [rbp+0x2220], rdx
       mov      dword ptr [rbp+0x2278], edi
       jmp      G_M000_IG10
 
G_M000_IG19:                ;; offset=0x0DDC
       mov      r10d, r15d
       imul     r10d, dword ptr [rbp+0x2268]
       mov      dword ptr [rbp-0x3C], r10d
 
G_M000_IG20:                ;; offset=0x0DEB
       cmp      r8d, edi
       jge      G_M000_IG31
 
G_M000_IG21:                ;; offset=0x0DF4
       vxorps   ymm0, ymm0, ymm0
       vxorps   ymm1, ymm1, ymm1
       vxorps   ymm2, ymm2, ymm2
       vxorps   ymm3, ymm3, ymm3
       mov      ebx, eax
       imul     ebx, dword ptr [rbp+0x21E4]
       lea      ebx, [rbx+8*rbx]
       movsxd   rbx, ebx
       mov      r12, qword ptr [rbp+0x2218]
       lea      rbx, [r12+4*rbx]
       mov      r12, qword ptr [rbp-0x38]
       lea      r9, [rbx+r12]
       lea      rsi, [r9+r12]
       add      r12, rsi
       mov      dword ptr [rbp+0x21E0], r15d
       mov      dword ptr [rbp+0x2278], edi
       mov      r10d, r15d
       imul     r10d, edi
       add      r10d, r8d
       cmp      r10d, dword ptr [rbp+0x21F4]
       setl     r10b
       movzx    r10, r10b
       mov      dword ptr [rbp+0x113C], r10d
       xor      r11d, r11d
       mov      dword ptr [rbp+0x21DC], r8d
       mov      r15d, r8d
       imul     r15d, dword ptr [rbp+0x2268]
       mov      dword ptr [rbp-0x40], r15d
       cmp      r11d, eax
       jge      G_M000_IG29
 
G_M000_IG22:                ;; offset=0x0E7C
       xor      r15d, r15d
       mov      edi, r11d
       sar      edi, 31
       and      edi, 15
       add      edi, r11d
       sar      edi, 4
       imul     edi, dword ptr [rbp+0x2200]
       add      edi, dword ptr [rbp-0x3C]
       mov      dword ptr [rbp-0x2C], edi
 
G_M000_IG23:                ;; offset=0x0E9B
       add      edi, r15d
       imul     edi, dword ptr [rbp+0x21FC]
       add      edi, dword ptr [rbp-0x40]
       shl      edi, 4
       mov      r14d, 3
 
G_M000_IG24:                ;; offset=0x0EB1
       mov      r8d, r11d
       sar      r8d, 31
       and      r8d, 15
       add      r8d, r11d
       and      r8d, -16
       mov      r10d, r11d
       sub      r10d, r8d
       add      r10d, edi
       movsxd   r8, r10d
       vbroadcastss zmm4, dword ptr [rdx+4*r8]
       vmovups  zmm5, zmmword ptr [rbx]
       vmovups  zmm6, zmmword ptr [r9]
       vmovups  zmm7, zmmword ptr [rsi]
       vmovups  zmm8, zmmword ptr [r12]
       mov      r10d, dword ptr [rbp+0x113C]
       test     r10d, r10d
       je       G_M000_IG30
 
G_M000_IG25:                ;; offset=0x0EFF
       vfmadd231ps zmm0, zmm5, zmm4
       vfmadd231ps zmm1, zmm6, zmm4
       vfmadd231ps zmm2, zmm7, zmm4
       vfmadd231ps zmm3, zmm8, zmm4
 
G_M000_IG26:                ;; offset=0x0F17
       add      rbx, 64
       add      r9, 64
       add      rsi, 64
       add      r12, 64
       add      edi, 16
       dec      r14d
       jne      SHORT G_M000_IG24
 
G_M000_IG27:                ;; offset=0x0F2F
       inc      r15d
       cmp      r15d, 3
       mov      edi, dword ptr [rbp-0x2C]
       jl       G_M000_IG23
 
G_M000_IG28:                ;; offset=0x0F3F
       inc      r11d
       cmp      r11d, eax
       jl       G_M000_IG22
 
G_M000_IG29:                ;; offset=0x0F4B
       mov      r11d, dword ptr [rbp+0x21E4]
       mov      edi, r11d
       sar      edi, 31
       and      edi, 15
       add      edi, r11d
       sar      edi, 4
       mov      r14d, dword ptr [rbp+0x21F8]
       mov      esi, edi
       imul     esi, r14d
       mov      r15d, dword ptr [rbp+0x21E0]
       mov      r9d, dword ptr [rbp+0x2278]
       mov      r10d, r15d
       imul     r10d, r9d
       add      esi, r10d
       mov      r8d, dword ptr [rbp+0x21DC]
       add      esi, r8d
       shl      esi, 4
       movsxd   rsi, esi
       vmovups  zmmword ptr [rcx+4*rsi], zmm0
       lea      esi, [rdi+0x01]
       imul     esi, r14d
       mov      r10d, r15d
       imul     r10d, r9d
       add      esi, r10d
       add      esi, r8d
       shl      esi, 4
       movsxd   rsi, esi
       vmovups  zmmword ptr [rcx+4*rsi], zmm1
       lea      esi, [rdi+0x02]
       imul     esi, r14d
       mov      r10d, r15d
       imul     r10d, r9d
       add      esi, r10d
       add      esi, r8d
       shl      esi, 4
       movsxd   rsi, esi
       vmovups  zmmword ptr [rcx+4*rsi], zmm2
       add      edi, 3
       mov      dword ptr [rbp+0x21F8], r14d
       imul     edi, r14d
       mov      dword ptr [rbp+0x2278], r9d
       mov      esi, r15d
       imul     esi, r9d
       add      edi, esi
       add      edi, r8d
       shl      edi, 4
       movsxd   rdi, edi
       mov      qword ptr [rbp+0x2210], rcx
       vmovups  zmmword ptr [rcx+4*rdi], zmm3
       inc      r8d
       mov      dword ptr [rbp+0x21E4], r11d
       mov      rcx, qword ptr [rbp+0x2210]
       mov      edi, dword ptr [rbp+0x2278]
       jmp      G_M000_IG20
 
G_M000_IG30:                ;; offset=0x1030
       vmulps   zmm9, zmm4, zmm5
       vaddps   zmm0, zmm9, zmm0
       vmulps   zmm10, zmm4, zmm6
       vaddps   zmm1, zmm10, zmm1
       vmulps   zmm11, zmm4, zmm7
       vaddps   zmm2, zmm11, zmm2
       vmulps   zmm12, zmm4, zmm8
       vaddps   zmm3, zmm12, zmm3
       jmp      G_M000_IG26
 
G_M000_IG31:                ;; offset=0x1065
       inc      r15d
       mov      r10d, dword ptr [rbp+0x2270]
       cmp      r15d, r10d
       jge      SHORT G_M000_IG34
 
G_M000_IG32:                ;; offset=0x1074
       xor      ebx, ebx
       mov      dword ptr [rbp+0x2270], r10d
       mov      r8d, ebx
       jmp      G_M000_IG06
 
G_M000_IG33:                ;; offset=0x1085
       xor      r15d, r15d
       mov      dword ptr [rbp+0x2270], r10d
       test     r10d, r10d
       mov      dword ptr [rbp+0x21E4], r11d
       mov      dword ptr [rbp+0x2208], r8d
       mov      r10d, dword ptr [rbp+0x2270]
       jg       SHORT G_M000_IG32
 
G_M000_IG34:                ;; offset=0x10A9
       mov      r11d, dword ptr [rbp+0x21E4]
       add      r11d, 64
       mov      r8d, dword ptr [rbp+0x2208]
       cmp      r11d, r8d
       jl       SHORT G_M000_IG33
 
G_M000_IG35:                ;; offset=0x10C0
       vzeroupper 
       add      rsp, 0x2268
       pop      rbx
       pop      r12
       pop      r13
       pop      r14
       pop      r15
       pop      rbp
       ret      
 
; Total bytes of code 4309

