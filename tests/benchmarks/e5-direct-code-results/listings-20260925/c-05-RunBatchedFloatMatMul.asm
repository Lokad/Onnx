; Assembly listing for method Lokad.Onnx.Tensor`1[float]:RunBatchedFloatMatMul(Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],Lokad.Onnx.TensorExecutionOptions) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 118772
; 73 inlinees with PGO data; 166 single block inlinees; 3 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       55                   push     rbp
       4157                 push     r15
       4156                 push     r14
       4155                 push     r13
       4154                 push     r12
       53                   push     rbx
       4881ECD8030000       sub      rsp, 984
       488DAC2400040000     lea      rbp, [rsp+0x400]
       C4413857C0           vxorps   xmm8, xmm8, xmm8
       62717D087F45D2       vmovdqa32 xmmword ptr [rbp-0x2E0], xmm8
       48B860FDFFFFFFFFFFFF mov      rax, -672
       C5797F4405D0         vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       C5797F4405E0         vmovdqa  xmmword ptr [rbp+rax-0x20], xmm8
       C5797F4405F0         vmovdqa  xmmword ptr [rbp+rax-0x10], xmm8
       4883C030             add      rax, 48
       75E8                 jne      SHORT  -5 instr
       488945D0             mov      qword ptr [rbp-0x30], rax
       4C8BFF               mov      r15, rdi
       488BDE               mov      rbx, rsi
       4C8BF2               mov      r14, rdx
 
G_M000_IG02:                ;; offset=0x0054
       48BF002FBD57B67F0000 mov      rdi, 0x7FB657BD2F00
       E8FDC8F77C           call     CORINFO_HELP_NEWSFAST
       48898548FDFFFF       mov      gword ptr [rbp-0x2B8], rax
       488D7848             lea      rdi, bword ptr [rax+0x48]
       40383F               cmp      byte  ptr [rdi], dil
       488D7510             lea      rsi, [rbp+0x10]
       BA30000000           mov      edx, 48
       C5F877               vzeroupper 
       FF1525779AFE         call     [CORINFO_HELP_BULK_WRITEBARRIER]
 
G_M000_IG03:                ;; offset=0x0083
       488B8548FDFFFF       mov      rax, gword ptr [rbp-0x2B8]
       C5FE6F4048           vmovdqu  ymm0, ymmword ptr [rax+0x48]
       C5FE7F85D8FEFFFF     vmovdqu  ymmword ptr [rbp-0x128], ymm0
       C5FA6F4068           vmovdqu  xmm0, xmmword ptr [rax+0x68]
       C5FA7F85F8FEFFFF     vmovdqu  xmmword ptr [rbp-0x108], xmm0
 
G_M000_IG04:                ;; offset=0x00A4
       498B7F10             mov      rdi, gword ptr [r15+0x10]
       488BCF               mov      rcx, rdi
       4885C9               test     rcx, rcx
       0F84A01B0000         je       G_M000_IG255
       488D5110             lea      rdx, bword ptr [rcx+0x10]
       8B4908               mov      ecx, dword ptr [rcx+0x08]
 
G_M000_IG05:                ;; offset=0x00BB
       8D71FE               lea      esi, [rcx-0x02]
       3BF1                 cmp      esi, ecx
       0F838A280000         jae      G_M000_IG380
       8BCE                 mov      ecx, esi
       448B2C8A             mov      r13d, dword ptr [rdx+4*rcx]
       4885FF               test     rdi, rdi
       0F84881B0000         je       G_M000_IG256
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG06:                ;; offset=0x00DC
       8D57FF               lea      edx, [rdi-0x01]
       3BD7                 cmp      edx, edi
       0F8369280000         jae      G_M000_IG380
       8BFA                 mov      edi, edx
       448B24B9             mov      r12d, dword ptr [rcx+4*rdi]
       488B7B10             mov      rdi, gword ptr [rbx+0x10]
       4885FF               test     rdi, rdi
       0F846C1B0000         je       G_M000_IG257
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG07:                ;; offset=0x0101
       8D57FF               lea      edx, [rdi-0x01]
       3BD7                 cmp      edx, edi
       0F8344280000         jae      G_M000_IG380
       8BFA                 mov      edi, edx
       8B0CB9               mov      ecx, dword ptr [rcx+4*rdi]
       898D0CFFFFFF         mov      dword ptr [rbp-0xF4], ecx
       8BBD00FFFFFF         mov      edi, dword ptr [rbp-0x100]
       0FB69504FFFFFF       movzx    rdx, byte  ptr [rbp-0xFC]
       400FB6B505FFFFFF     movzx    rsi, byte  ptr [rbp-0xFB]
       85D2                 test     edx, edx
       7413                 je       SHORT G_M000_IG09
 
G_M000_IG08:                ;; offset=0x0130
       85F6                 test     esi, esi
       740F                 je       SHORT G_M000_IG09
       83FF01               cmp      edi, 1
       750A                 jne      SHORT G_M000_IG09
       4183FD30             cmp      r13d, 48
       0F8D2C1B0000         jge      G_M000_IG258
 
G_M000_IG09:                ;; offset=0x0143
       33FF                 xor      rdi, rdi
       33D2                 xor      rdx, rdx
 
G_M000_IG10:                ;; offset=0x0147
       488995E0FCFFFF       mov      gword ptr [rbp-0x320], rdx
       4885D2               test     rdx, rdx
       0F85831B0000         jne      G_M000_IG260
       33F6                 xor      esi, esi
 
G_M000_IG11:                ;; offset=0x0159
       33FF                 xor      rdi, rdi
       4889BD78FEFFFF       mov      gword ptr [rbp-0x188], rdi
       85F6                 test     esi, esi
       0F8514090000         jne      G_M000_IG116
       488B8548FDFFFF       mov      rax, gword ptr [rbp-0x2B8]
       4C8B6850             mov      r13, gword ptr [rax+0x50]
       498BFF               mov      rdi, r15
       48BEB8669C57B67F0000 mov      rsi, 0x7FB6579C66B8
       483937               cmp      qword ptr [rdi], rsi
       0F853C200000         jne      G_M000_IG293
       33FF                 xor      rdi, rdi
 
G_M000_IG12:                ;; offset=0x018D
       4885FF               test     rdi, rdi
       0F854B200000         jne      G_M000_IG294
 
G_M000_IG13:                ;; offset=0x0196
       4D8BE7               mov      r12, r15
       48BFB8669C57B67F0000 mov      rdi, 0x7FB6579C66B8
       49393C24             cmp      qword ptr [r12], rdi
       0F8547200000         jne      G_M000_IG295
 
G_M000_IG14:                ;; offset=0x01AD
       4D85E4               test     r12, r12
       0F84AE210000         je       G_M000_IG311
       41807C243C00         cmp      byte  ptr [r12+0x3C], 0
       0F85A2210000         jne      G_M000_IG311
       498B7C2418           mov      rdi, gword ptr [r12+0x18]
       4885FF               test     rdi, rdi
       0F8434200000         je       G_M000_IG296
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       448B4708             mov      r8d, dword ptr [rdi+0x08]
 
G_M000_IG15:                ;; offset=0x01D8
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       4885FF               test     rdi, rdi
       0F8444200000         je       G_M000_IG297
       488D7710             lea      rsi, bword ptr [rdi+0x10]
       8B5708               mov      edx, dword ptr [rdi+0x08]
       48898D70FCFFFF       mov      bword ptr [rbp-0x390], rcx
       44898580FDFFFF       mov      dword ptr [rbp-0x280], r8d
 
G_M000_IG16:                ;; offset=0x01FB
       488BFE               mov      rdi, rsi
       8BF2                 mov      esi, edx
       33D2                 xor      edx, edx
       FF15286361FF         call     [Lokad.Onnx.ArrayUtilities:GetStrides(System.ReadOnlySpan`1[int],bool):int[]]
       4885C0               test     rax, rax
       0F8430200000         je       G_M000_IG298
       488D7810             lea      rdi, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG17:                ;; offset=0x0218
       398580FDFFFF         cmp      dword ptr [rbp-0x280], eax
       0F85D9200000         jne      G_M000_IG309
       488B8D70FCFFFF       mov      rcx, bword ptr [rbp-0x390]
       488BD1               mov      rdx, rcx
       488BF7               mov      rsi, rdi
       8BC0                 mov      eax, eax
       48C1E002             shl      rax, 2
       4883F808             cmp      rax, 8
       0F8209200000         jb       G_M000_IG299
 
G_M000_IG18:                ;; offset=0x0241
       483BD6               cmp      rdx, rsi
       7456                 je       SHORT G_M000_IG24
 
G_M000_IG19:                ;; offset=0x0246
       4883F840             cmp      rax, 64
       0F82D1160000         jb       G_M000_IG201
 
G_M000_IG20:                ;; offset=0x0250
       33C9                 xor      ecx, ecx
       4883C0C0             add      rax, -64
       7428                 je       SHORT G_M000_IG23
 
G_M000_IG21:                ;; offset=0x0258
       62F17C481002         vmovups  zmm0, zmmword ptr [rdx]
       62F37D483E0E04       vpcmpfalseub k1, zmm0, zmmword ptr [rsi]
       C4E1F898C9           kortestq k1, k1
       0F8563170000         jne      G_M000_IG213
 
G_M000_IG22:                ;; offset=0x0273
       4883C140             add      rcx, 64
       483BC1               cmp      rax, rcx
       0F871D200000         ja       G_M000_IG306
 
G_M000_IG23:                ;; offset=0x0280
       62F17C48100402       vmovups  zmm0, zmmword ptr [rdx+rax]
       62F17D48740C06       vpcmpeqb k1, zmm0, zmmword ptr [rsi+rax]
       C4E1F898C9           kortestq k1, k1
       0F833A170000         jae      G_M000_IG213
 
G_M000_IG24:                ;; offset=0x029C
       BF01000000           mov      edi, 1
 
G_M000_IG25:                ;; offset=0x02A1
       85FF                 test     edi, edi
       0F84BB200000         je       G_M000_IG311
       418B7C245C           mov      edi, dword ptr [r12+0x5C]
       413B7C2430           cmp      edi, dword ptr [r12+0x30]
       0F854B200000         jne      G_M000_IG310
 
G_M000_IG26:                ;; offset=0x02B9
       4D8BFC               mov      r15, r12
 
G_M000_IG27:                ;; offset=0x02BC
       488B8548FDFFFF       mov      rax, gword ptr [rbp-0x2B8]
       4C8B6850             mov      r13, gword ptr [rax+0x50]
       488BFB               mov      rdi, rbx
       48BEB8669C57B67F0000 mov      rsi, 0x7FB6579C66B8
       483937               cmp      qword ptr [rdi], rsi
       0F85A8200000         jne      G_M000_IG312
       33FF                 xor      rdi, rdi
 
G_M000_IG28:                ;; offset=0x02DF
       4885FF               test     rdi, rdi
       0F85B7200000         jne      G_M000_IG313
 
G_M000_IG29:                ;; offset=0x02E8
       4C8BE3               mov      r12, rbx
       48BFB8669C57B67F0000 mov      rdi, 0x7FB6579C66B8
       49393C24             cmp      qword ptr [r12], rdi
       0F85B3200000         jne      G_M000_IG314
 
G_M000_IG30:                ;; offset=0x02FF
       4D85E4               test     r12, r12
       0F841A220000         je       G_M000_IG330
       41807C243C00         cmp      byte  ptr [r12+0x3C], 0
       0F850E220000         jne      G_M000_IG330
       498B7C2418           mov      rdi, gword ptr [r12+0x18]
       4885FF               test     rdi, rdi
       0F84A0200000         je       G_M000_IG315
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       448B4708             mov      r8d, dword ptr [rdi+0x08]
 
G_M000_IG31:                ;; offset=0x032A
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       4885FF               test     rdi, rdi
       0F84B0200000         je       G_M000_IG316
       488D7710             lea      rsi, bword ptr [rdi+0x10]
       8B5708               mov      edx, dword ptr [rdi+0x08]
       48898D68FCFFFF       mov      bword ptr [rbp-0x398], rcx
       4489857CFDFFFF       mov      dword ptr [rbp-0x284], r8d
 
G_M000_IG32:                ;; offset=0x034D
       488BFE               mov      rdi, rsi
       8BF2                 mov      esi, edx
       33D2                 xor      edx, edx
       FF15D66161FF         call     [Lokad.Onnx.ArrayUtilities:GetStrides(System.ReadOnlySpan`1[int],bool):int[]]
       4885C0               test     rax, rax
       0F849C200000         je       G_M000_IG317
       488D7810             lea      rdi, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG33:                ;; offset=0x036A
       39857CFDFFFF         cmp      dword ptr [rbp-0x284], eax
       0F8545210000         jne      G_M000_IG328
       488B8D68FCFFFF       mov      rcx, bword ptr [rbp-0x398]
       488BD1               mov      rdx, rcx
       488BF7               mov      rsi, rdi
       8BC0                 mov      eax, eax
       48C1E002             shl      rax, 2
       4883F808             cmp      rax, 8
       0F8275200000         jb       G_M000_IG318
 
G_M000_IG34:                ;; offset=0x0393
       483BD6               cmp      rdx, rsi
       7456                 je       SHORT G_M000_IG40
 
G_M000_IG35:                ;; offset=0x0398
       4883F840             cmp      rax, 64
       0F823B160000         jb       G_M000_IG214
 
G_M000_IG36:                ;; offset=0x03A2
       33C9                 xor      ecx, ecx
       4883C0C0             add      rax, -64
       7428                 je       SHORT G_M000_IG39
 
G_M000_IG37:                ;; offset=0x03AA
       62F17C481002         vmovups  zmm0, zmmword ptr [rdx]
       62F37D483E0E04       vpcmpfalseub k1, zmm0, zmmword ptr [rsi]
       C4E1F898C9           kortestq k1, k1
       0F85CD160000         jne      G_M000_IG226
 
G_M000_IG38:                ;; offset=0x03C5
       4883C140             add      rcx, 64
       483BC1               cmp      rax, rcx
       0F8789200000         ja       G_M000_IG325
 
G_M000_IG39:                ;; offset=0x03D2
       62F17C48100402       vmovups  zmm0, zmmword ptr [rdx+rax]
       62F17D48740C06       vpcmpeqb k1, zmm0, zmmword ptr [rsi+rax]
       C4E1F898C9           kortestq k1, k1
       0F83A4160000         jae      G_M000_IG226
 
G_M000_IG40:                ;; offset=0x03EE
       BF01000000           mov      edi, 1
 
G_M000_IG41:                ;; offset=0x03F3
       85FF                 test     edi, edi
       0F8427210000         je       G_M000_IG330
       418B7C245C           mov      edi, dword ptr [r12+0x5C]
       413B7C2430           cmp      edi, dword ptr [r12+0x30]
       0F85B7200000         jne      G_M000_IG329
 
G_M000_IG42:                ;; offset=0x040B
       498BDC               mov      rbx, r12
 
G_M000_IG43:                ;; offset=0x040E
       488B8548FDFFFF       mov      rax, gword ptr [rbp-0x2B8]
       4C8B6850             mov      r13, gword ptr [rax+0x50]
       4D8BE6               mov      r12, r14
       4D85E4               test     r12, r12
       7414                 je       SHORT G_M000_IG45
 
G_M000_IG44:                ;; offset=0x0421
       48BFB8669C57B67F0000 mov      rdi, 0x7FB6579C66B8
       49393C24             cmp      qword ptr [r12], rdi
       0F850E210000         jne      G_M000_IG331
 
G_M000_IG45:                ;; offset=0x0435
       4D85E4               test     r12, r12
       0F8475220000         je       G_M000_IG347
       41807C243C00         cmp      byte  ptr [r12+0x3C], 0
       0F8569220000         jne      G_M000_IG347
       498B7C2418           mov      rdi, gword ptr [r12+0x18]
       4885FF               test     rdi, rdi
       0F84FB200000         je       G_M000_IG332
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       448B4708             mov      r8d, dword ptr [rdi+0x08]
 
G_M000_IG46:                ;; offset=0x0460
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       4885FF               test     rdi, rdi
       0F840B210000         je       G_M000_IG333
       488D7710             lea      rsi, bword ptr [rdi+0x10]
       8B5708               mov      edx, dword ptr [rdi+0x08]
       48898D60FCFFFF       mov      bword ptr [rbp-0x3A0], rcx
       44898578FDFFFF       mov      dword ptr [rbp-0x288], r8d
 
G_M000_IG47:                ;; offset=0x0483
       488BFE               mov      rdi, rsi
       8BF2                 mov      esi, edx
       33D2                 xor      edx, edx
       FF15A06061FF         call     [Lokad.Onnx.ArrayUtilities:GetStrides(System.ReadOnlySpan`1[int],bool):int[]]
       4885C0               test     rax, rax
       0F84F7200000         je       G_M000_IG334
       488D7810             lea      rdi, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG48:                ;; offset=0x04A0
       398578FDFFFF         cmp      dword ptr [rbp-0x288], eax
       0F85A0210000         jne      G_M000_IG345
       488B8D60FCFFFF       mov      rcx, bword ptr [rbp-0x3A0]
       488BD1               mov      rdx, rcx
       488BF7               mov      rsi, rdi
       8BC0                 mov      eax, eax
       48C1E002             shl      rax, 2
       4883F808             cmp      rax, 8
       0F82D0200000         jb       G_M000_IG335
 
G_M000_IG49:                ;; offset=0x04C9
       483BD6               cmp      rdx, rsi
       7456                 je       SHORT G_M000_IG55
 
G_M000_IG50:                ;; offset=0x04CE
       4883F840             cmp      rax, 64
       0F82C1150000         jb       G_M000_IG227
 
G_M000_IG51:                ;; offset=0x04D8
       33C9                 xor      ecx, ecx
       4883C0C0             add      rax, -64
       7428                 je       SHORT G_M000_IG54
 
G_M000_IG52:                ;; offset=0x04E0
       62F17C481002         vmovups  zmm0, zmmword ptr [rdx]
       62F37D483E0E04       vpcmpfalseub k1, zmm0, zmmword ptr [rsi]
       C4E1F898C9           kortestq k1, k1
       0F8553160000         jne      G_M000_IG239
 
G_M000_IG53:                ;; offset=0x04FB
       4883C140             add      rcx, 64
       483BC1               cmp      rax, rcx
       0F87E4200000         ja       G_M000_IG342
 
G_M000_IG54:                ;; offset=0x0508
       62F17C48100402       vmovups  zmm0, zmmword ptr [rdx+rax]
       62F17D48740C06       vpcmpeqb k1, zmm0, zmmword ptr [rsi+rax]
       C4E1F898C9           kortestq k1, k1
       0F832A160000         jae      G_M000_IG239
 
G_M000_IG55:                ;; offset=0x0524
       BF01000000           mov      edi, 1
 
G_M000_IG56:                ;; offset=0x0529
       85FF                 test     edi, edi
       0F8482210000         je       G_M000_IG347
       418B7C245C           mov      edi, dword ptr [r12+0x5C]
       413B7C2430           cmp      edi, dword ptr [r12+0x30]
       0F8512210000         jne      G_M000_IG346
 
G_M000_IG57:                ;; offset=0x0541
       4D8BF4               mov      r14, r12
       498B7710             mov      rsi, gword ptr [r15+0x10]
       4885F6               test     rsi, rsi
       0F8483210000         je       G_M000_IG348
       488D7E10             lea      rdi, bword ptr [rsi+0x10]
       8B7608               mov      esi, dword ptr [rsi+0x08]
 
G_M000_IG58:                ;; offset=0x0558
       4889BD78FFFFFF       mov      bword ptr [rbp-0x88], rdi
       897580               mov      dword ptr [rbp-0x80], esi
       448B6D80             mov      r13d, dword ptr [rbp-0x80]
       4183C5FE             add      r13d, -2
       443B6D80             cmp      r13d, dword ptr [rbp-0x80]
       0F8769210000         ja       G_M000_IG349
       4C8BA578FFFFFF       mov      r12, bword ptr [rbp-0x88]
       4585ED               test     r13d, r13d
       0F8460210000         je       G_M000_IG350
       4963F5               movsxd   rsi, r13d
       48BF58B18556B67F0000 mov      rdi, 0x7FB65685B158
       E8DAC4F77C           call     CORINFO_HELP_NEWARR_1_VC
       488BF8               mov      rdi, rax
       4889BDC8FCFFFF       mov      gword ptr [rbp-0x338], rdi
       418BD5               mov      edx, r13d
       48C1E202             shl      rdx, 2
       4883C710             add      rdi, 16
       498BF4               mov      rsi, r12
       FF156C729AFE         call     [System.SpanHelpers:Memmove(byref,byref,nuint)]
       4C8BADC8FCFFFF       mov      r13, gword ptr [rbp-0x338]
 
G_M000_IG59:                ;; offset=0x05BB
       4C89AD40FDFFFF       mov      gword ptr [rbp-0x2C0], r13
       498B7710             mov      rsi, gword ptr [r15+0x10]
       488BFE               mov      rdi, rsi
       4885FF               test     rdi, rdi
       0F8421210000         je       G_M000_IG351
       488D4710             lea      rax, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG60:                ;; offset=0x05D9
       48898578FFFFFF       mov      bword ptr [rbp-0x88], rax
       897D80               mov      dword ptr [rbp-0x80], edi
       8B7D80               mov      edi, dword ptr [rbp-0x80]
       83C7FE               add      edi, -2
       3B7D80               cmp      edi, dword ptr [rbp-0x80]
       0F835E230000         jae      G_M000_IG380
       488B8578FFFFFF       mov      rax, bword ptr [rbp-0x88]
       448B2CB8             mov      r13d, dword ptr [rax+4*rdi]
       488B8548FDFFFF       mov      rax, gword ptr [rbp-0x2B8]
       44896838             mov      dword ptr [rax+0x38], r13d
       488BFE               mov      rdi, rsi
       4885FF               test     rdi, rdi
       0F84E8200000         je       G_M000_IG352
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG61:                ;; offset=0x061B
       48898D78FFFFFF       mov      bword ptr [rbp-0x88], rcx
       897D80               mov      dword ptr [rbp-0x80], edi
       8B7D80               mov      edi, dword ptr [rbp-0x80]
       FFCF                 dec      edi
       3B7D80               cmp      edi, dword ptr [rbp-0x80]
       0F831D230000         jae      G_M000_IG380
       488B8D78FFFFFF       mov      rcx, bword ptr [rbp-0x88]
       8B3CB9               mov      edi, dword ptr [rcx+4*rdi]
       89783C               mov      dword ptr [rax+0x3C], edi
       488B7B10             mov      rdi, gword ptr [rbx+0x10]
       4885FF               test     rdi, rdi
       0F84B8200000         je       G_M000_IG353
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG62:                ;; offset=0x0654
       48898D78FFFFFF       mov      bword ptr [rbp-0x88], rcx
       897D80               mov      dword ptr [rbp-0x80], edi
       8B7D80               mov      edi, dword ptr [rbp-0x80]
       FFCF                 dec      edi
       3B7D80               cmp      edi, dword ptr [rbp-0x80]
       0F83E4220000         jae      G_M000_IG380
       488B8D78FFFFFF       mov      rcx, bword ptr [rbp-0x88]
       8B3CB9               mov      edi, dword ptr [rcx+4*rdi]
       897840               mov      dword ptr [rax+0x40], edi
       488BBD40FDFFFF       mov      rdi, gword ptr [rbp-0x2C0]
       448B6708             mov      r12d, dword ptr [rdi+0x08]
       4489A574FDFFFF       mov      dword ptr [rbp-0x28C], r12d
       4885F6               test     rsi, rsi
       0F847A200000         je       G_M000_IG354
       488D5610             lea      rdx, bword ptr [rsi+0x10]
       448B4608             mov      r8d, dword ptr [rsi+0x08]
 
G_M000_IG63:                ;; offset=0x069C
       4D8BCF               mov      r9, r15
       48BEB8669C57B67F0000 mov      rsi, 0x7FB6579C66B8
       493931               cmp      qword ptr [r9], rsi
       0F8582200000         jne      G_M000_IG355
       4533C9               xor      r9, r9
 
G_M000_IG64:                ;; offset=0x06B5
       4D85C9               test     r9, r9
       0F85AC200000         jne      G_M000_IG356
 
G_M000_IG65:                ;; offset=0x06BE
       4D8B5718             mov      r10, gword ptr [r15+0x18]
       4C8995C0FCFFFF       mov      gword ptr [rbp-0x340], r10
 
G_M000_IG66:                ;; offset=0x06C9
       8B8D74FDFFFF         mov      ecx, dword ptr [rbp-0x28C]
       898D6CFDFFFF         mov      dword ptr [rbp-0x294], ecx
       48899550FCFFFF       mov      bword ptr [rbp-0x3B0], rdx
       44898568FDFFFF       mov      dword ptr [rbp-0x298], r8d
       8BF1                 mov      esi, ecx
       48BF58B18556B67F0000 mov      rdi, 0x7FB65685B158
       E87CC3F77C           call     CORINFO_HELP_NEWARR_1_VC
       488BC8               mov      rcx, rax
       33F6                 xor      esi, esi
       8BBD6CFDFFFF         mov      edi, dword ptr [rbp-0x294]
       85FF                 test     edi, edi
       7E5F                 jle      SHORT G_M000_IG76
 
G_M000_IG67:                ;; offset=0x0703
       4C8B95C0FCFFFF       mov      r10, gword ptr [rbp-0x340]
       4D85D2               test     r10, r10
       0F844E140000         je       G_M000_IG241
 
G_M000_IG68:                ;; offset=0x0713
       448B8568FDFFFF       mov      r8d, dword ptr [rbp-0x298]
       413BF8               cmp      edi, r8d
       0F8F6C200000         jg       G_M000_IG359
 
G_M000_IG69:                ;; offset=0x0723
       41397A08             cmp      dword ptr [r10+0x08], edi
       0F8C5D200000         jl       G_M000_IG358
 
G_M000_IG70:                ;; offset=0x072D
       397908               cmp      dword ptr [rcx+0x08], edi
       0F8C4F200000         jl       G_M000_IG357
 
G_M000_IG71:                ;; offset=0x0736
       4183C0FE             add      r8d, -2
 
G_M000_IG72:                ;; offset=0x073A
       413BF0               cmp      esi, r8d
       7D13                 jge      SHORT G_M000_IG74
 
G_M000_IG73:                ;; offset=0x073F
       8BC6                 mov      eax, esi
       488B9550FCFFFF       mov      rdx, bword ptr [rbp-0x3B0]
       833C8201             cmp      dword ptr [rdx+4*rax], 1
       0F8503140000         jne      G_M000_IG240
 
G_M000_IG74:                ;; offset=0x0752
       33C0                 xor      eax, eax
 
G_M000_IG75:                ;; offset=0x0754
       448BCE               mov      r9d, esi
       4289448910           mov      dword ptr [rcx+4*r9+0x10], eax
       FFC6                 inc      esi
       3BF7                 cmp      esi, edi
       7CD8                 jl       SHORT G_M000_IG72
 
G_M000_IG76:                ;; offset=0x0762
       48898D38FDFFFF       mov      gword ptr [rbp-0x2C8], rcx
       4489A564FDFFFF       mov      dword ptr [rbp-0x29C], r12d
       488B7310             mov      rsi, gword ptr [rbx+0x10]
       4885F6               test     rsi, rsi
       0F842D200000         je       G_M000_IG361
       488D4E10             lea      rcx, bword ptr [rsi+0x10]
       8B5608               mov      edx, dword ptr [rsi+0x08]
 
G_M000_IG77:                ;; offset=0x0784
       4C8BC3               mov      r8, rbx
       4D85C0               test     r8, r8
       7416                 je       SHORT G_M000_IG79
 
G_M000_IG78:                ;; offset=0x078C
       48BEB8669C57B67F0000 mov      rsi, 0x7FB6579C66B8
       493930               cmp      qword ptr [r8], rsi
       0F852E200000         jne      G_M000_IG362
       4533C0               xor      r8, r8
 
G_M000_IG79:                ;; offset=0x07A2
       4D85C0               test     r8, r8
       0F8556200000         jne      G_M000_IG363
 
G_M000_IG80:                ;; offset=0x07AB
       4C8B4B18             mov      r9, gword ptr [rbx+0x18]
       4C898DB8FCFFFF       mov      gword ptr [rbp-0x348], r9
 
G_M000_IG81:                ;; offset=0x07B6
       8B8564FDFFFF         mov      eax, dword ptr [rbp-0x29C]
       89855CFDFFFF         mov      dword ptr [rbp-0x2A4], eax
       48898D40FCFFFF       mov      bword ptr [rbp-0x3C0], rcx
       899558FDFFFF         mov      dword ptr [rbp-0x2A8], edx
       8BF0                 mov      esi, eax
       48BF58B18556B67F0000 mov      rdi, 0x7FB65685B158
       E890C2F77C           call     CORINFO_HELP_NEWARR_1_VC
       488BF8               mov      rdi, rax
       33F6                 xor      esi, esi
       8B855CFDFFFF         mov      eax, dword ptr [rbp-0x2A4]
       85C0                 test     eax, eax
       7E5C                 jle      SHORT G_M000_IG91
 
G_M000_IG82:                ;; offset=0x07EF
       4C8B8DB8FCFFFF       mov      r9, gword ptr [rbp-0x348]
       4D85C9               test     r9, r9
       0F84B8130000         je       G_M000_IG246
 
G_M000_IG83:                ;; offset=0x07FF
       8B9558FDFFFF         mov      edx, dword ptr [rbp-0x2A8]
       3BC2                 cmp      eax, edx
       0F8F19200000         jg       G_M000_IG366
 
G_M000_IG84:                ;; offset=0x080D
       41394108             cmp      dword ptr [r9+0x08], eax
       0F8C0A200000         jl       G_M000_IG365
 
G_M000_IG85:                ;; offset=0x0817
       394708               cmp      dword ptr [rdi+0x08], eax
       0F8CFC1F0000         jl       G_M000_IG364
 
G_M000_IG86:                ;; offset=0x0820
       83C2FE               add      edx, -2
 
G_M000_IG87:                ;; offset=0x0823
       3BF2                 cmp      esi, edx
       7D14                 jge      SHORT G_M000_IG89
 
G_M000_IG88:                ;; offset=0x0827
       8BCE                 mov      ecx, esi
       4C8B8540FCFFFF       mov      r8, bword ptr [rbp-0x3C0]
       41833C8801           cmp      dword ptr [r8+4*rcx], 1
       0F8570130000         jne      G_M000_IG245
 
G_M000_IG89:                ;; offset=0x083B
       33C9                 xor      ecx, ecx
 
G_M000_IG90:                ;; offset=0x083D
       448BD6               mov      r10d, esi
       42894C9710           mov      dword ptr [rdi+4*r10+0x10], ecx
       FFC6                 inc      esi
       3BF0                 cmp      esi, eax
       7CD8                 jl       SHORT G_M000_IG87
 
G_M000_IG91:                ;; offset=0x084B
       4889BD30FDFFFF       mov      gword ptr [rbp-0x2D0], rdi
       418BC4               mov      eax, r12d
       498B7610             mov      rsi, gword ptr [r14+0x10]
       4885F6               test     rsi, rsi
       0F84DF1F0000         je       G_M000_IG368
       488D4E10             lea      rcx, bword ptr [rsi+0x10]
       8B5608               mov      edx, dword ptr [rsi+0x08]
 
G_M000_IG92:                ;; offset=0x0869
       898554FDFFFF         mov      dword ptr [rbp-0x2AC], eax
       48898D38FCFFFF       mov      bword ptr [rbp-0x3C8], rcx
       899550FDFFFF         mov      dword ptr [rbp-0x2B0], edx
       4D8B4618             mov      r8, gword ptr [r14+0x18]
       4C8985B0FCFFFF       mov      gword ptr [rbp-0x350], r8
       8BF0                 mov      esi, eax
       48BF58B18556B67F0000 mov      rdi, 0x7FB65685B158
       E8D8C1F77C           call     CORINFO_HELP_NEWARR_1_VC
       4C8BC0               mov      r8, rax
       33C0                 xor      eax, eax
       8B9554FDFFFF         mov      edx, dword ptr [rbp-0x2AC]
       85D2                 test     edx, edx
       7E5C                 jle      SHORT G_M000_IG102
 
G_M000_IG93:                ;; offset=0x08A7
       488BBDB0FCFFFF       mov      rdi, gword ptr [rbp-0x350]
       4885FF               test     rdi, rdi
       0F8453130000         je       G_M000_IG251
 
G_M000_IG94:                ;; offset=0x08B7
       8BB550FDFFFF         mov      esi, dword ptr [rbp-0x2B0]
       3BD6                 cmp      edx, esi
       0F8F8F1F0000         jg       G_M000_IG371
 
G_M000_IG95:                ;; offset=0x08C5
       395708               cmp      dword ptr [rdi+0x08], edx
       0F8C811F0000         jl       G_M000_IG370
 
G_M000_IG96:                ;; offset=0x08CE
       41395008             cmp      dword ptr [r8+0x08], edx
       0F8C721F0000         jl       G_M000_IG369
 
G_M000_IG97:                ;; offset=0x08D8
       83C6FE               add      esi, -2
 
G_M000_IG98:                ;; offset=0x08DB
       3BC6                 cmp      eax, esi
       7D14                 jge      SHORT G_M000_IG100
 
G_M000_IG99:                ;; offset=0x08DF
       8BC8                 mov      ecx, eax
       4C8B8D38FCFFFF       mov      r9, bword ptr [rbp-0x3C8]
       41833C8901           cmp      dword ptr [r9+4*rcx], 1
       0F850C130000         jne      G_M000_IG250
 
G_M000_IG100:                ;; offset=0x08F3
       33C9                 xor      ecx, ecx
 
G_M000_IG101:                ;; offset=0x08F5
       448BD0               mov      r10d, eax
       43894C9010           mov      dword ptr [r8+4*r10+0x10], ecx
       FFC0                 inc      eax
       3BC2                 cmp      eax, edx
       7CD8                 jl       SHORT G_M000_IG98
 
G_M000_IG102:                ;; offset=0x0903
       4C898528FDFFFF       mov      gword ptr [rbp-0x2D8], r8
       488BBD40FDFFFF       mov      rdi, gword ptr [rbp-0x2C0]
       488D4710             lea      rax, bword ptr [rdi+0x10]
       418BD4               mov      edx, r12d
       BE01000000           mov      esi, 1
       85D2                 test     edx, edx
       7E0E                 jle      SHORT G_M000_IG105
 
G_M000_IG103:                ;; offset=0x0921
       33C9                 xor      ecx, ecx
 
G_M000_IG104:                ;; offset=0x0923
       0FAF3408             imul     esi, dword ptr [rax+rcx]
       4883C104             add      rcx, 4
       FFCA                 dec      edx
       75F4                 jne      SHORT G_M000_IG104
 
G_M000_IG105:                ;; offset=0x092F
       89B544FEFFFF         mov      dword ptr [rbp-0x1BC], esi
       8975D4               mov      dword ptr [rbp-0x2C], esi
       488B8548FDFFFF       mov      rax, gword ptr [rbp-0x2B8]
       83787002             cmp      dword ptr [rax+0x70], 2
       0F8D241F0000         jge      G_M000_IG373
 
G_M000_IG106:                ;; offset=0x0949
       C78538FFFFFF01000000 mov      dword ptr [rbp-0xC8], 1
 
G_M000_IG107:                ;; offset=0x0953
       488D5048             lea      rdx, bword ptr [rax+0x48]
       4C8B4A10             mov      r9, gword ptr [rdx+0x10]
       440FB6522C           movzx    r10, byte  ptr [rdx+0x2C]
       0FB6522D             movzx    rdx, byte  ptr [rdx+0x2D]
       4585D2               test     r10d, r10d
       0F8427010000         je       G_M000_IG118
       85D2                 test     edx, edx
       0F841F010000         je       G_M000_IG118
       41F6C501             test     r13b, 1
       0F85111F0000         jne      G_M000_IG374
 
G_M000_IG108:                ;; offset=0x097F
       498BF9               mov      rdi, r9
       488BF3               mov      rsi, rbx
       FF15F5E2CBFF         call     [Lokad.Onnx.GraphPacking:ResolvePacked(System.Collections.Generic.IReadOnlyDictionary`2[float[],Lokad.Onnx.PackedMatMulWeight],Lokad.Onnx.Tensor`1[float]):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE8               mov      r13, rax
       4D85ED               test     r13, r13
       0F84FD000000         je       G_M000_IG118
 
G_M000_IG109:                ;; offset=0x0997
       498B7D10             mov      rdi, gword ptr [r13+0x10]
       488BC7               mov      rax, rdi
       4885C0               test     rax, rax
       0F840B1F0000         je       G_M000_IG375
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG110:                ;; offset=0x09AA
       83F802               cmp      eax, 2
       0F85E1000000         jne      G_M000_IG118
       488BC7               mov      rax, rdi
       4885C0               test     rax, rax
       0F84FA1E0000         je       G_M000_IG376
       488D4810             lea      rcx, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG111:                ;; offset=0x09C6
       85C0                 test     eax, eax
       0F84821F0000         je       G_M000_IG380
       8B01                 mov      eax, dword ptr [rcx]
       4885FF               test     rdi, rdi
       0F84E91E0000         je       G_M000_IG377
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG112:                ;; offset=0x09E0
       83FF01               cmp      edi, 1
       0F86671F0000         jbe      G_M000_IG380
       8B7904               mov      edi, dword ptr [rcx+0x04]
       85C0                 test     eax, eax
       0F8EA0000000         jle      G_M000_IG118
       85FF                 test     edi, edi
       0F8E98000000         jle      G_M000_IG118
       3D00100000           cmp      eax, 0x1000
       0F8D8D000000         jge      G_M000_IG118
       4898                 cdqe     
       4863FF               movsxd   rdi, edi
       480FAFF8             imul     rdi, rax
       4881FF00000008       cmp      rdi, 0x8000000
       7F7B                 jg       SHORT G_M000_IG118
 
G_M000_IG113:                ;; offset=0x0A19
       4D85ED               test     r13, r13
       747B                 je       SHORT G_M000_IG119
 
G_M000_IG114:                ;; offset=0x0A22
       837DD401             cmp      dword ptr [rbp-0x2C], 1
       0F859F1E0000         jne      G_M000_IG378
 
G_M000_IG115:                ;; offset=0x0A2C
       8B8D38FFFFFF         mov      ecx, dword ptr [rbp-0xC8]
       890C24               mov      dword ptr [rsp], ecx
       488B8548FDFFFF       mov      rax, gword ptr [rbp-0x2B8]
       8B5038               mov      edx, dword ptr [rax+0x38]
       89542408             mov      dword ptr [rsp+0x08], edx
       8B503C               mov      edx, dword ptr [rax+0x3C]
       89542410             mov      dword ptr [rsp+0x10], edx
       8B5040               mov      edx, dword ptr [rax+0x40]
       89542418             mov      dword ptr [rsp+0x18], edx
       4C896C2420           mov      gword ptr [rsp+0x20], r13
       498BFF               mov      rdi, r15
       498BF6               mov      rsi, r14
       488B9540FDFFFF       mov      rdx, gword ptr [rbp-0x2C0]
       488B8D38FDFFFF       mov      rcx, gword ptr [rbp-0x2C8]
       4C8B8528FDFFFF       mov      r8, gword ptr [rbp-0x2D8]
       448B8D44FEFFFF       mov      r9d, dword ptr [rbp-0x1BC]
       FF154AF7CBFF         call     [Lokad.Onnx.Tensor`1[float]:RunPackedBatches(Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],int[],int[],int[],int,int,int,int,int,Lokad.Onnx.DenseTensor`1[float])]
 
G_M000_IG116:                ;; offset=0x0A7E
       90                   nop      
 
G_M000_IG117:                ;; offset=0x0A7F
       C5F877               vzeroupper 
       4881C4D8030000       add      rsp, 984
       5B                   pop      rbx
       415C                 pop      r12
       415D                 pop      r13
       415E                 pop      r14
       415F                 pop      r15
       5D                   pop      rbp
       C3                   ret      
 
G_M000_IG118:                ;; offset=0x0A94
       4533ED               xor      r13, r13
       EB80                 jmp      SHORT G_M000_IG113
                            align    [0 bytes for IG133]
 
G_M000_IG119:                ;; offset=0x0A99
       498BFF               mov      rdi, r15
       FF15DEF3CBFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       48898568FFFFFF       mov      gword ptr [rbp-0x98], rax
       48899570FFFFFF       mov      qword ptr [rbp-0x90], rdx
       488DBD68FFFFFF       lea      rdi, [rbp-0x98]
       488D75B8             lea      rsi, [rbp-0x48]
       FF15EF79B9FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG120:                ;; offset=0x0AC2
       488BFB               mov      rdi, rbx
       393F                 cmp      dword ptr [rdi], edi
       FF15B3F3CBFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       48898568FFFFFF       mov      gword ptr [rbp-0x98], rax
       48899570FFFFFF       mov      qword ptr [rbp-0x90], rdx
       488DBD68FFFFFF       lea      rdi, [rbp-0x98]
       488D75A0             lea      rsi, [rbp-0x60]
       FF15C479B9FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG121:                ;; offset=0x0AED
       498BFE               mov      rdi, r14
       393F                 cmp      dword ptr [rdi], edi
       FF1588F3CBFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       48898568FFFFFF       mov      gword ptr [rbp-0x98], rax
       48899570FFFFFF       mov      qword ptr [rbp-0x90], rdx
       488DBD68FFFFFF       lea      rdi, [rbp-0x98]
       488D7588             lea      rsi, [rbp-0x78]
       FF159979B9FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG122:                ;; offset=0x0B18
       488B75C0             mov      rsi, qword ptr [rbp-0x40]
       488B8548FDFFFF       mov      rax, gword ptr [rbp-0x2B8]
       48897020             mov      qword ptr [rax+0x20], rsi
       488B75A8             mov      rsi, qword ptr [rbp-0x58]
       48897028             mov      qword ptr [rax+0x28], rsi
       488B7590             mov      rsi, qword ptr [rbp-0x70]
       48897030             mov      qword ptr [rax+0x30], rsi
       8B9D38FFFFFF         mov      ebx, dword ptr [rbp-0xC8]
       83FB01               cmp      ebx, 1
       0F8F4A060000         jg       G_M000_IG167
 
G_M000_IG123:                ;; offset=0x0B46
       488B7020             mov      rsi, qword ptr [rax+0x20]
       4889B560FFFFFF       mov      qword ptr [rbp-0xA0], rsi
       488B7028             mov      rsi, qword ptr [rax+0x28]
       4889B558FFFFFF       mov      qword ptr [rbp-0xA8], rsi
       488B7030             mov      rsi, qword ptr [rax+0x30]
       4889B550FFFFFF       mov      qword ptr [rbp-0xB0], rsi
       4489A54CFFFFFF       mov      dword ptr [rbp-0xB4], r12d
       8BB54CFFFFFF         mov      esi, dword ptr [rbp-0xB4]
       48BF58B18556B67F0000 mov      rdi, 0x7FB65685B158
       E8EDBEF77C           call     CORINFO_HELP_NEWARR_1_VC
       48898520FDFFFF       mov      gword ptr [rbp-0x2E0], rax
       33C9                 xor      ecx, ecx
       898D48FFFFFF         mov      dword ptr [rbp-0xB8], ecx
 
G_M000_IG124:                ;; offset=0x0B92
       898D44FFFFFF         mov      dword ptr [rbp-0xBC], ecx
 
G_M000_IG125:                ;; offset=0x0B98
       898D40FFFFFF         mov      dword ptr [rbp-0xC0], ecx
 
G_M000_IG126:                ;; offset=0x0B9E
       898D3CFFFFFF         mov      dword ptr [rbp-0xC4], ecx
       448B45D4             mov      r8d, dword ptr [rbp-0x2C]
       4439853CFFFFFF       cmp      dword ptr [rbp-0xC4], r8d
       0F8C4E010000         jl       G_M000_IG135
       E9420D0000           jmp      G_M000_IG198
 
G_M000_IG127:                ;; offset=0x0BBA
       E8281E0000           call     G_M000_IG396
       90                   nop      
 
G_M000_IG128:                ;; offset=0x0BC0
       BA56555555           mov      edx, 0x55555556
       8BC2                 mov      eax, edx
       F7AD40FEFFFF         imul     edx:eax, dword ptr [rbp-0x1C0]
       448BCA               mov      r9d, edx
       41C1E91F             shr      r9d, 31
       4403CA               add      r9d, edx
       478D0C49             lea      r9d, [r9+2*r9]
       448BB540FEFFFF       mov      r14d, dword ptr [rbp-0x1C0]
       452BF1               sub      r14d, r9d
       0F84690B0000         je       G_M000_IG191
 
G_M000_IG129:                ;; offset=0x0BEB
       4533C9               xor      r9d, r9d
 
G_M000_IG130:                ;; offset=0x0BEE
       8B8DBCFDFFFF         mov      ecx, dword ptr [rbp-0x244]
       3B8D40FEFFFF         cmp      ecx, dword ptr [rbp-0x1C0]
       0F85AD0B0000         jne      G_M000_IG193
 
G_M000_IG131:                ;; offset=0x0C00
       33FF                 xor      rdi, rdi
       4889BDB0FDFFFF       mov      gword ptr [rbp-0x250], rdi
 
G_M000_IG132:                ;; offset=0x0C09
       448BA54CFFFFFF       mov      r12d, dword ptr [rbp-0xB4]
       418D7C24FF           lea      edi, [r12-0x01]
       85FF                 test     edi, edi
       0F8CA4000000         jl       G_M000_IG134
 
G_M000_IG133:                ;; offset=0x0C1D
       488B8D20FDFFFF       mov      rcx, gword ptr [rbp-0x2E0]
       8B7108               mov      esi, dword ptr [rcx+0x08]
       8BDE                 mov      ebx, esi
       3BFB                 cmp      edi, ebx
       0F83C50C0000         jae      G_M000_IG197
       488D74B910           lea      rsi, bword ptr [rcx+4*rdi+0x10]
       FF06                 inc      dword ptr [rsi]
       488B8538FDFFFF       mov      rax, gword ptr [rbp-0x2C8]
       3B7808               cmp      edi, dword ptr [rax+0x08]
       0F83AE0C0000         jae      G_M000_IG197
       8B74B810             mov      esi, dword ptr [rax+4*rdi+0x10]
       03B548FFFFFF         add      esi, dword ptr [rbp-0xB8]
       89B548FFFFFF         mov      dword ptr [rbp-0xB8], esi
       4C8B8530FDFFFF       mov      r8, gword ptr [rbp-0x2D0]
       413B7808             cmp      edi, dword ptr [r8+0x08]
       0F838D0C0000         jae      G_M000_IG197
       418B74B810           mov      esi, dword ptr [r8+4*rdi+0x10]
       03B544FFFFFF         add      esi, dword ptr [rbp-0xBC]
       89B544FFFFFF         mov      dword ptr [rbp-0xBC], esi
       4C8B8D28FDFFFF       mov      r9, gword ptr [rbp-0x2D8]
       413B7908             cmp      edi, dword ptr [r9+0x08]
       0F836B0C0000         jae      G_M000_IG197
       418B74B910           mov      esi, dword ptr [r9+4*rdi+0x10]
       03B540FFFFFF         add      esi, dword ptr [rbp-0xC0]
       89B540FFFFFF         mov      dword ptr [rbp-0xC0], esi
       3BFB                 cmp      edi, ebx
       0F83520C0000         jae      G_M000_IG197
       8B74B910             mov      esi, dword ptr [rcx+4*rdi+0x10]
       3BFB                 cmp      edi, ebx
       0F83460C0000         jae      G_M000_IG197
       488B9540FDFFFF       mov      rdx, gword ptr [rbp-0x2C0]
       3B74BA10             cmp      esi, dword ptr [rdx+4*rdi+0x10]
       0F8D3F040000         jge      G_M000_IG165
 
G_M000_IG134:                ;; offset=0x0CC1
       488B8D20FDFFFF       mov      rcx, gword ptr [rbp-0x2E0]
       488B8538FDFFFF       mov      rax, gword ptr [rbp-0x2C8]
       4C8B8D28FDFFFF       mov      r9, gword ptr [rbp-0x2D8]
       4C8B8530FDFFFF       mov      r8, gword ptr [rbp-0x2D0]
       488B9540FDFFFF       mov      rdx, gword ptr [rbp-0x2C0]
       8BBD3CFFFFFF         mov      edi, dword ptr [rbp-0xC4]
       FFC7                 inc      edi
       89BD3CFFFFFF         mov      dword ptr [rbp-0xC4], edi
       448B55D4             mov      r10d, dword ptr [rbp-0x2C]
       4439953CFFFFFF       cmp      dword ptr [rbp-0xC4], r10d
       0F8DF90B0000         jge      G_M000_IG198
 
G_M000_IG135:                ;; offset=0x0D03
       488B8548FDFFFF       mov      rax, gword ptr [rbp-0x2B8]
       8B4838               mov      ecx, dword ptr [rax+0x38]
       898D40FEFFFF         mov      dword ptr [rbp-0x1C0], ecx
       8B483C               mov      ecx, dword ptr [rax+0x3C]
       898D3CFEFFFF         mov      dword ptr [rbp-0x1C4], ecx
       8B4840               mov      ecx, dword ptr [rax+0x40]
       898D38FEFFFF         mov      dword ptr [rbp-0x1C8], ecx
       48638D48FFFFFF       movsxd   rcx, dword ptr [rbp-0xB8]
       488BB560FFFFFF       mov      rsi, qword ptr [rbp-0xA0]
       488D0C8E             lea      rcx, [rsi+4*rcx]
       48898D30FEFFFF       mov      qword ptr [rbp-0x1D0], rcx
       48638D44FFFFFF       movsxd   rcx, dword ptr [rbp-0xBC]
       4C8B9558FFFFFF       mov      r10, qword ptr [rbp-0xA8]
       498D0C8A             lea      rcx, [r10+4*rcx]
       48898D28FEFFFF       mov      qword ptr [rbp-0x1D8], rcx
       48638D40FFFFFF       movsxd   rcx, dword ptr [rbp-0xC0]
       4C8B9D50FFFFFF       mov      r11, qword ptr [rbp-0xB0]
       498D0C8B             lea      rcx, [r11+4*rcx]
       48898D20FEFFFF       mov      qword ptr [rbp-0x1E0], rcx
       488D4848             lea      rcx, bword ptr [rax+0x48]
       4C8B21               mov      r12, gword ptr [rcx]
       0FB6592C             movzx    rbx, byte  ptr [rcx+0x2C]
       440FB6792D           movzx    r15, byte  ptr [rcx+0x2D]
 
G_M000_IG136:                ;; offset=0x0D80
       C5FE6F01             vmovdqu  ymm0, ymmword ptr [rcx]
       C5FE7F85F0FDFFFF     vmovdqu  ymmword ptr [rbp-0x210], ymm0
       C5FA6F4120           vmovdqu  xmm0, xmmword ptr [rcx+0x20]
       62F17E087F45E1       vmovdqu32 xmmword ptr [rbp-0x1F0], xmm0
 
G_M000_IG137:                ;; offset=0x0D98
       83BD40FEFFFF30       cmp      dword ptr [rbp-0x1C0], 48
       0F8D15050000         jge      G_M000_IG169
 
G_M000_IG138:                ;; offset=0x0DA5
       C5FE6F85F0FDFFFF     vmovdqu  ymm0, ymmword ptr [rbp-0x210]
       62F17E287F45EE       vmovdqu32 ymmword ptr [rbp-0x240], ymm0
       62F17E086F45E1       vmovdqu32 xmm0, xmmword ptr [rbp-0x1F0]
       62F17E087F45DE       vmovdqu32 xmmword ptr [rbp-0x220], xmm0
 
G_M000_IG139:                ;; offset=0x0DC2
       4533C9               xor      r9, r9
       4C898DB0FDFFFF       mov      gword ptr [rbp-0x250], r9
       85DB                 test     ebx, ebx
       0F84FA020000         je       G_M000_IG164
 
G_M000_IG140:                ;; offset=0x0DD4
       4585FF               test     r15d, r15d
       740D                 je       SHORT G_M000_IG141
       83BD40FEFFFF01       cmp      dword ptr [rbp-0x1C0], 1
       0F84C6050000         je       G_M000_IG170
 
G_M000_IG141:                ;; offset=0x0DE6
       4585FF               test     r15d, r15d
       0F84220A0000         je       G_M000_IG194
       83BD40FEFFFF02       cmp      dword ptr [rbp-0x1C0], 2
       0F8C150A0000         jl       G_M000_IG194
       8B9540FEFFFF         mov      edx, dword ptr [rbp-0x1C0]
       8B8D40FEFFFF         mov      ecx, dword ptr [rbp-0x1C0]
       8BBD40FEFFFF         mov      edi, dword ptr [rbp-0x1C0]
       C1EF1F               shr      edi, 31
       03BD40FEFFFF         add      edi, dword ptr [rbp-0x1C0]
       83E7FE               and      edi, -2
       2BCF                 sub      ecx, edi
       2BD1                 sub      edx, ecx
       8995BCFDFFFF         mov      dword ptr [rbp-0x244], edx
       BA56555555           mov      edx, 0x55555556
       8BC2                 mov      eax, edx
       F7AD40FEFFFF         imul     edx:eax, dword ptr [rbp-0x1C0]
       8BCA                 mov      ecx, edx
       C1E91F               shr      ecx, 31
       03CA                 add      ecx, edx
       8D0C49               lea      ecx, [rcx+2*rcx]
       448BB540FEFFFF       mov      r14d, dword ptr [rbp-0x1C0]
       442BF1               sub      r14d, ecx
       0F84A2050000         je       G_M000_IG171
 
G_M000_IG142:                ;; offset=0x0E4B
       83BDBCFDFFFF40       cmp      dword ptr [rbp-0x244], 64
       0F8D03070000         jge      G_M000_IG179
 
G_M000_IG143:                ;; offset=0x0E58
       4585F6               test     r14d, r14d
       740D                 je       SHORT G_M000_IG145
 
G_M000_IG144:                ;; offset=0x0E5D
       F68540FEFFFF01       test     byte  ptr [rbp-0x1C0], 1
       0F856E080000         jne      G_M000_IG188
 
G_M000_IG145:                ;; offset=0x0E6A
       4863BD3CFEFFFF       movsxd   rdi, dword ptr [rbp-0x1C4]
       48638538FEFFFF       movsxd   rax, dword ptr [rbp-0x1C8]
       480FAFF8             imul     rdi, rax
       4881FF00000100       cmp      rdi, 0x10000
       0F8F4F080000         jg       G_M000_IG188
 
G_M000_IG146:                ;; offset=0x0E89
       448BBD3CFEFFFF       mov      r15d, dword ptr [rbp-0x1C4]
       440FAFBD38FEFFFF     imul     r15d, dword ptr [rbp-0x1C8]
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       488B3F               mov      rdi, gword ptr [rdi]
       418BF7               mov      esi, r15d
       FF1582C8D5FF         call     [System.Buffers.SharedArrayPool`1[float]:Rent(int):float[]:this]
       488BD8               mov      rbx, rax
       4D85E4               test     r12, r12
       7424                 je       SHORT G_M000_IG147
       48BE7083AF57B67F0000 mov      rsi, 0x7FB657AF8370
       49393424             cmp      qword ptr [r12], rsi
       0F85F2070000         jne      G_M000_IG187
       4983C408             add      r12, 8
       4963F7               movsxd   rsi, r15d
       48C1E602             shl      rsi, 2
       F0                   lock     
       49013424             add      qword ptr [r12], rsi
 
G_M000_IG147:                ;; offset=0x0EDA
       48899DA8FCFFFF       mov      gword ptr [rbp-0x358], rbx
 
G_M000_IG148:                ;; offset=0x0EE1
       48899DB0FDFFFF       mov      gword ptr [rbp-0x250], rbx
       4885DB               test     rbx, rbx
       0F84E8000000         je       G_M000_IG156
 
G_M000_IG149:                ;; offset=0x0EF1
       837B0800             cmp      dword ptr [rbx+0x08], 0
       0F84DE000000         je       G_M000_IG156
       837B0800             cmp      dword ptr [rbx+0x08], 0
       0F86B5010000         jbe      G_M000_IG162
       4883C310             add      rbx, 16
       4C8BC3               mov      r8, rbx
 
G_M000_IG150:                ;; offset=0x0F0C
       8BBD38FEFFFF         mov      edi, dword ptr [rbp-0x1C8]
       8BB538FEFFFF         mov      esi, dword ptr [rbp-0x1C8]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       C1FA1F               sar      edx, 31
       83E21F               and      edx, 31
       039538FEFFFF         add      edx, dword ptr [rbp-0x1C8]
       83E2E0               and      edx, -32
       2BF2                 sub      esi, edx
       2BFE                 sub      edi, esi
       33F6                 xor      esi, esi
       85FF                 test     edi, edi
       0F8FA6000000         jg       G_M000_IG157
 
G_M000_IG151:                ;; offset=0x0F3B
       8BB538FEFFFF         mov      esi, dword ptr [rbp-0x1C8]
       2BF7                 sub      esi, edi
       8BD7                 mov      edx, edi
       C1FA1F               sar      edx, 31
       83E21F               and      edx, 31
       03D7                 add      edx, edi
       C1FA05               sar      edx, 5
       0FAF953CFEFFFF       imul     edx, dword ptr [rbp-0x1C4]
       C1E205               shl      edx, 5
       4863D2               movsxd   rdx, edx
       498D1490             lea      rdx, [r8+4*rdx]
       33C9                 xor      ecx, ecx
       83BD3CFEFFFF00       cmp      dword ptr [rbp-0x1C4], 0
       0F8EF5000000         jle      G_M000_IG160
 
G_M000_IG152:                ;; offset=0x0F70
       4863FF               movsxd   rdi, edi
       48C1E702             shl      rdi, 2
       EB19                 jmp      SHORT G_M000_IG154
       6666660F1F840000000000 align    [11 bytes for IG155]
 
G_M000_IG153:                ;; offset=0x0F84
       FFC1                 inc      ecx
       3B8D3CFEFFFF         cmp      ecx, dword ptr [rbp-0x1C4]
       0F8DD3000000         jge      G_M000_IG160
 
G_M000_IG154:                ;; offset=0x0F92
       448BC9               mov      r9d, ecx
       440FAF8D38FEFFFF     imul     r9d, dword ptr [rbp-0x1C8]
       4D63C9               movsxd   r9, r9d
       49C1E102             shl      r9, 2
       4C038D28FEFFFF       add      r9, qword ptr [rbp-0x1D8]
       4C03CF               add      r9, rdi
       8BC1                 mov      eax, ecx
       0FAFC6               imul     eax, esi
       4898                 cdqe     
       488D0482             lea      rax, [rdx+4*rax]
       4533D2               xor      r10d, r10d
       85F6                 test     esi, esi
       7EC4                 jle      SHORT G_M000_IG153
 
G_M000_IG155:                ;; offset=0x0FC0
       4D63DA               movsxd   r11, r10d
       C4817A100499         vmovss   xmm0, dword ptr [r9+4*r11]
       C4A17A110498         vmovss   dword ptr [rax+4*r11], xmm0
       41FFC2               inc      r10d
       443BD6               cmp      r10d, esi
       7CE9                 jl       SHORT G_M000_IG155
       EBAB                 jmp      SHORT G_M000_IG153
 
G_M000_IG156:                ;; offset=0x0FD9
       4533C0               xor      r8d, r8d
       E92BFFFFFF           jmp      G_M000_IG150
                            align    [0 bytes for IG158]
 
G_M000_IG157:                ;; offset=0x0FE1
       8BD6                 mov      edx, esi
       C1FA1F               sar      edx, 31
       83E21F               and      edx, 31
       03D6                 add      edx, esi
       C1FA05               sar      edx, 5
       0FAF953CFEFFFF       imul     edx, dword ptr [rbp-0x1C4]
       C1E205               shl      edx, 5
       4863D2               movsxd   rdx, edx
       498D1490             lea      rdx, [r8+4*rdx]
       33C9                 xor      ecx, ecx
       83BD3CFEFFFF00       cmp      dword ptr [rbp-0x1C4], 0
       7E4F                 jle      SHORT G_M000_IG159
 
G_M000_IG158:                ;; offset=0x100A
       448BC9               mov      r9d, ecx
       41C1E105             shl      r9d, 5
       4D63C9               movsxd   r9, r9d
       4E8D0C8A             lea      r9, [rdx+4*r9]
       8BC1                 mov      eax, ecx
       0FAF8538FEFFFF       imul     eax, dword ptr [rbp-0x1C8]
       4898                 cdqe     
       48C1E002             shl      rax, 2
       48038528FEFFFF       add      rax, qword ptr [rbp-0x1D8]
       4C63D6               movsxd   r10, esi
       4A8D0490             lea      rax, [rax+4*r10]
       62F17E486F00         vmovdqu32 zmm0, zmmword ptr [rax]
       62F17E486F4801       vmovdqu32 zmm1, zmmword ptr [rax+0x40]
       62D17E487F01         vmovdqu32 zmmword ptr [r9], zmm0
       62D17E487F4901       vmovdqu32 zmmword ptr [r9+0x40], zmm1
       FFC1                 inc      ecx
       3B8D3CFEFFFF         cmp      ecx, dword ptr [rbp-0x1C4]
       7CB1                 jl       SHORT G_M000_IG158
 
G_M000_IG159:                ;; offset=0x1059
       83C620               add      esi, 32
       3BF7                 cmp      esi, edi
       7C81                 jl       SHORT G_M000_IG157
       E9D6FEFFFF           jmp      G_M000_IG151
 
G_M000_IG160:                ;; offset=0x1065
       4585F6               test     r14d, r14d
       7428                 je       SHORT G_M000_IG161
       8BBD40FEFFFF         mov      edi, dword ptr [rbp-0x1C0]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       FF15B8F2CBFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       EB2E                 jmp      SHORT G_M000_IG163
 
G_M000_IG161:                ;; offset=0x1092
       8BBD40FEFFFF         mov      edi, dword ptr [rbp-0x1C0]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       FF1578F2CBFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       EB06                 jmp      SHORT G_M000_IG163
 
G_M000_IG162:                ;; offset=0x10BA
       E8D1269AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG163:                ;; offset=0x10C0
       33FF                 xor      rdi, rdi
       4889BDB0FDFFFF       mov      gword ptr [rbp-0x250], rdi
       E9ECFAFFFF           jmp      G_M000_IG127
 
G_M000_IG164:                ;; offset=0x10CE
       8BBD40FEFFFF         mov      edi, dword ptr [rbp-0x1C0]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4C8B8528FEFFFF       mov      r8, qword ptr [rbp-0x1D8]
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       FF1585FCCBFF         call     [Lokad.Onnx.MathOps:mm(int,int,int,ptr,ptr,ptr)]
       E900FBFFFF           jmp      G_M000_IG131
 
G_M000_IG165:                ;; offset=0x1100
       85FF                 test     edi, edi
       0F8C72070000         jl       G_M000_IG196
       3BFB                 cmp      edi, ebx
       0F8D6A070000         jge      G_M000_IG196
       33F6                 xor      esi, esi
       8974B910             mov      dword ptr [rcx+4*rdi+0x10], esi
       3B7808               cmp      edi, dword ptr [rax+0x08]
       0F83D7070000         jae      G_M000_IG197
       8B74B810             mov      esi, dword ptr [rax+4*rdi+0x10]
       0FAF74BA10           imul     esi, dword ptr [rdx+4*rdi+0x10]
       448B9548FFFFFF       mov      r10d, dword ptr [rbp-0xB8]
       442BD6               sub      r10d, esi
       44899548FFFFFF       mov      dword ptr [rbp-0xB8], r10d
       413B7808             cmp      edi, dword ptr [r8+0x08]
       0F83B3070000         jae      G_M000_IG197
       418B74B810           mov      esi, dword ptr [r8+4*rdi+0x10]
       0FAF74BA10           imul     esi, dword ptr [rdx+4*rdi+0x10]
       448B9544FFFFFF       mov      r10d, dword ptr [rbp-0xBC]
       442BD6               sub      r10d, esi
       44899544FFFFFF       mov      dword ptr [rbp-0xBC], r10d
       413B7908             cmp      edi, dword ptr [r9+0x08]
       0F838E070000         jae      G_M000_IG197
       418B74B910           mov      esi, dword ptr [r9+4*rdi+0x10]
       0FAF74BA10           imul     esi, dword ptr [rdx+4*rdi+0x10]
       448B9540FFFFFF       mov      r10d, dword ptr [rbp-0xC0]
       442BD6               sub      r10d, esi
       44899540FFFFFF       mov      dword ptr [rbp-0xC0], r10d
 
G_M000_IG166:                ;; offset=0x1183
       FFCF                 dec      edi
       0F8992FAFFFF         jns      G_M000_IG133
       E931FBFFFF           jmp      G_M000_IG134
 
G_M000_IG167:                ;; offset=0x1190
       448BBD44FEFFFF       mov      r15d, dword ptr [rbp-0x1BC]
       4963F7               movsxd   rsi, r15d
       48BF58B18556B67F0000 mov      rdi, 0x7FB65685B158
       E8C7B8F77C           call     CORINFO_HELP_NEWARR_1_VC
       488B8D48FDFFFF       mov      rcx, gword ptr [rbp-0x2B8]
       488D7908             lea      rdi, bword ptr [rcx+0x08]
       488BF0               mov      rsi, rax
       E8A40E9AFD           call     CORINFO_HELP_ASSIGN_REF
       4963F7               movsxd   rsi, r15d
       48BF58B18556B67F0000 mov      rdi, 0x7FB65685B158
       E8A2B8F77C           call     CORINFO_HELP_NEWARR_1_VC
       488B8D48FDFFFF       mov      rcx, gword ptr [rbp-0x2B8]
       488D7910             lea      rdi, bword ptr [rcx+0x10]
       488BF0               mov      rsi, rax
       E87F0E9AFD           call     CORINFO_HELP_ASSIGN_REF
       4963F7               movsxd   rsi, r15d
       48BF58B18556B67F0000 mov      rdi, 0x7FB65685B158
       E87DB8F77C           call     CORINFO_HELP_NEWARR_1_VC
       488B8D48FDFFFF       mov      rcx, gword ptr [rbp-0x2B8]
       488D7918             lea      rdi, bword ptr [rcx+0x18]
       488BF0               mov      rsi, rax
       E85A0E9AFD           call     CORINFO_HELP_ASSIGN_REF
       488B9540FDFFFF       mov      rdx, gword ptr [rbp-0x2C0]
       488D7A10             lea      rdi, bword ptr [rdx+0x10]
       418BF4               mov      esi, r12d
       488B8D48FDFFFF       mov      rcx, gword ptr [rbp-0x2B8]
       4C8B4910             mov      r9, gword ptr [rcx+0x10]
       4C890C24             mov      gword ptr [rsp], r9
       4C8B4918             mov      r9, gword ptr [rcx+0x18]
       4C894C2408           mov      gword ptr [rsp+0x08], r9
       4C8B4908             mov      r9, gword ptr [rcx+0x08]
       488B9538FDFFFF       mov      rdx, gword ptr [rbp-0x2C8]
       488B8D30FDFFFF       mov      rcx, gword ptr [rbp-0x2D0]
       4C8B8528FDFFFF       mov      r8, gword ptr [rbp-0x2D8]
       FF1595EFCBFF         call     [Lokad.Onnx.Tensor`1[float]:FillBatchOffsets(System.ReadOnlySpan`1[int],int[],int[],int[],int[],int[],int[])]
       48BF2039BD57B67F0000 mov      rdi, 0x7FB657BD3920
       E806B7F77C           call     CORINFO_HELP_NEWSFAST
       4C8BF0               mov      r14, rax
       498BFE               mov      rdi, r14
 
G_M000_IG168:                ;; offset=0x1260
       FF1592EFCBFF         call     [System.Threading.Tasks.ParallelOptions:.ctor():this]
       498BFE               mov      rdi, r14
       8BF3                 mov      esi, ebx
       FF159FEFCBFF         call     [System.Threading.Tasks.ParallelOptions:set_MaxDegreeOfParallelism(int):this]
       48BF483ABD57B67F0000 mov      rdi, 0x7FB657BD3A48
       E8E0B6F77C           call     CORINFO_HELP_NEWSFAST
       488BD8               mov      rbx, rax
       488BFB               mov      rdi, rbx
       488BB548FDFFFF       mov      rsi, gword ptr [rbp-0x2B8]
       48BAE8A0B357B67F0000 mov      rdx, 0x7FB657B3A0E8
       FF158B7B9AFE         call     [System.MulticastDelegate:CtorClosed(System.Object,nint):this]
       488DBD20FFFFFF       lea      rdi, [rbp-0xE0]
       4C8BC3               mov      r8, rbx
       418BD7               mov      edx, r15d
       498BCE               mov      rcx, r14
       33F6                 xor      esi, esi
       FF1573EFCBFF         call     [System.Threading.Tasks.Parallel:For(int,int,System.Threading.Tasks.ParallelOptions,System.Action`1[int]):System.Threading.Tasks.ParallelLoopResult]
       E942060000           jmp      G_M000_IG198
 
G_M000_IG169:                ;; offset=0x12BA
       81BD3CFEFFFF00040000 cmp      dword ptr [rbp-0x1C4], 0x400
       0F8CDBFAFFFF         jl       G_M000_IG138
       81BD38FEFFFF00040000 cmp      dword ptr [rbp-0x1C8], 0x400
       0F8CCBFAFFFF         jl       G_M000_IG138
       48638D3CFEFFFF       movsxd   rcx, dword ptr [rbp-0x1C4]
       4863BD38FEFFFF       movsxd   rdi, dword ptr [rbp-0x1C8]
       480FAFCF             imul     rcx, rdi
       4881F900000004       cmp      rcx, 0x4000000
       0F8FACFAFFFF         jg       G_M000_IG138
       85DB                 test     ebx, ebx
       0F84A4FAFFFF         je       G_M000_IG138
       4585FF               test     r15d, r15d
       0F849BFAFFFF         je       G_M000_IG138
       4C89A5F0FDFFFF       mov      gword ptr [rbp-0x210], r12
       889D1CFEFFFF         mov      byte  ptr [rbp-0x1E4], bl
       4488BD1DFEFFFF       mov      byte  ptr [rbp-0x1E3], r15b
       488D3C24             lea      rdi, [rsp]
       488DB5F0FDFFFF       lea      rsi, [rbp-0x210]
       488B0E               mov      rcx, gword ptr [rsi]
       48890C24             mov      gword ptr [rsp], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2408           mov      gword ptr [rsp+0x08], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2410           mov      gword ptr [rsp+0x10], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2418           mov      gword ptr [rsp+0x18], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2420           mov      gword ptr [rsp+0x20], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       48A5                 movsq    
       8BBD40FEFFFF         mov      edi, dword ptr [rbp-0x1C0]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4C8B8528FEFFFF       mov      r8, qword ptr [rbp-0x1D8]
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       FF1501F9CBFF         call     [Lokad.Onnx.Tensor`1[float]:RunIsolatedShortWidePackedRows(int,int,int,ptr,ptr,ptr,Lokad.Onnx.TensorExecutionOptions)]
       E95DF8FFFF           jmp      G_M000_IG132
 
G_M000_IG170:                ;; offset=0x13AC
       81BD38FEFFFF00200000 cmp      dword ptr [rbp-0x1C8], 0x2000
       0F8C2AFAFFFF         jl       G_M000_IG141
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4C8B8528FEFFFF       mov      r8, qword ptr [rbp-0x1D8]
       BF01000000           mov      edi, 1
       FF1520F9CBFF         call     [Lokad.Onnx.MathOps:mm_m1_kblocked(int,int,int,ptr,ptr,ptr)]
       E913F8FFFF           jmp      G_M000_IG131
 
G_M000_IG171:                ;; offset=0x13ED
       83BD40FEFFFF40       cmp      dword ptr [rbp-0x1C0], 64
       0F8C51FAFFFF         jl       G_M000_IG142
       48638D3CFEFFFF       movsxd   rcx, dword ptr [rbp-0x1C4]
       4863BD38FEFFFF       movsxd   rdi, dword ptr [rbp-0x1C8]
       480FAFCF             imul     rcx, rdi
       4881F900000004       cmp      rcx, 0x4000000
       0F8F32FAFFFF         jg       G_M000_IG142
       4C89A5C0FDFFFF       mov      gword ptr [rbp-0x240], r12
       889DECFDFFFF         mov      byte  ptr [rbp-0x214], bl
       4488BDEDFDFFFF       mov      byte  ptr [rbp-0x213], r15b
       488D3C24             lea      rdi, [rsp]
       488DB5C0FDFFFF       lea      rsi, [rbp-0x240]
       488B0E               mov      rcx, gword ptr [rsi]
       48890C24             mov      gword ptr [rsp], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2408           mov      gword ptr [rsp+0x08], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2410           mov      gword ptr [rsp+0x10], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2418           mov      gword ptr [rsp+0x18], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2420           mov      gword ptr [rsp+0x20], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       48A5                 movsq    
       8BBD3CFEFFFF         mov      edi, dword ptr [rbp-0x1C4]
       0FAFBD38FEFFFF       imul     edi, dword ptr [rbp-0x1C8]
       FF153CF8CBFF         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       48898598FCFFFF       mov      gword ptr [rbp-0x368], rax
 
G_M000_IG172:                ;; offset=0x14A3
       488BBD98FCFFFF       mov      rdi, gword ptr [rbp-0x368]
       4889BDB0FDFFFF       mov      gword ptr [rbp-0x250], rdi
       4883BD98FCFFFF00     cmp      gword ptr [rbp-0x368], 0
       740D                 je       SHORT G_M000_IG173
       488BBD98FCFFFF       mov      rdi, gword ptr [rbp-0x368]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       750B                 jne      SHORT G_M000_IG175
 
G_M000_IG173:                ;; offset=0x14C8
       4533E4               xor      r12d, r12d
       EB1E                 jmp      SHORT G_M000_IG176
 
G_M000_IG174:                ;; offset=0x14CD
       E8BE229AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG175:                ;; offset=0x14D3
       488BBD98FCFFFF       mov      rdi, gword ptr [rbp-0x368]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       76ED                 jbe      SHORT G_M000_IG174
       4C8BA598FCFFFF       mov      r12, gword ptr [rbp-0x368]
       4983C410             add      r12, 16
 
G_M000_IG176:                ;; offset=0x14EB
       8BBD3CFEFFFF         mov      edi, dword ptr [rbp-0x1C4]
       8BB538FEFFFF         mov      esi, dword ptr [rbp-0x1C8]
       488B9528FEFFFF       mov      rdx, qword ptr [rbp-0x1D8]
       498BCC               mov      rcx, r12
       FF15D96FB9FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8BBD40FEFFFF         mov      edi, dword ptr [rbp-0x1C0]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4D8BC4               mov      r8, r12
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       FF1500EECBFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG177:                ;; offset=0x1531
       33FF                 xor      rdi, rdi
       4889BDB0FDFFFF       mov      gword ptr [rbp-0x250], rdi
 
G_M000_IG178:                ;; offset=0x153A
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB598FCFFFF       mov      rsi, gword ptr [rbp-0x368]
       33D2                 xor      edx, edx
       FF15E2C1D5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       E9F9010000           jmp      G_M000_IG191
 
G_M000_IG179:                ;; offset=0x155B
       48638D3CFEFFFF       movsxd   rcx, dword ptr [rbp-0x1C4]
       4863BD38FEFFFF       movsxd   rdi, dword ptr [rbp-0x1C8]
       480FAFCF             imul     rcx, rdi
       4881F900000004       cmp      rcx, 0x4000000
       0F8FDEF8FFFF         jg       G_M000_IG143
       4C89A5C0FDFFFF       mov      gword ptr [rbp-0x240], r12
       889DECFDFFFF         mov      byte  ptr [rbp-0x214], bl
       4488BDEDFDFFFF       mov      byte  ptr [rbp-0x213], r15b
       488D3C24             lea      rdi, [rsp]
       488DB5C0FDFFFF       lea      rsi, [rbp-0x240]
       488B0E               mov      rcx, gword ptr [rsi]
       48890C24             mov      gword ptr [rsp], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2408           mov      gword ptr [rsp+0x08], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2410           mov      gword ptr [rsp+0x10], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2418           mov      gword ptr [rsp+0x18], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       488B0E               mov      rcx, gword ptr [rsi]
       48894C2420           mov      gword ptr [rsp+0x20], rcx
       4883C608             add      rsi, 8
       4883C708             add      rdi, 8
       48A5                 movsq    
       8BBD3CFEFFFF         mov      edi, dword ptr [rbp-0x1C4]
       0FAFBD38FEFFFF       imul     edi, dword ptr [rbp-0x1C8]
       FF15DBF6CBFF         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488985A0FCFFFF       mov      gword ptr [rbp-0x360], rax
 
G_M000_IG180:                ;; offset=0x1604
       488BBDA0FCFFFF       mov      rdi, gword ptr [rbp-0x360]
       4889BDB0FDFFFF       mov      gword ptr [rbp-0x250], rdi
       4883BDA0FCFFFF00     cmp      gword ptr [rbp-0x360], 0
       740D                 je       SHORT G_M000_IG181
       488BBDA0FCFFFF       mov      rdi, gword ptr [rbp-0x360]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       750B                 jne      SHORT G_M000_IG183
 
G_M000_IG181:                ;; offset=0x1629
       4533F6               xor      r14d, r14d
       EB1E                 jmp      SHORT G_M000_IG184
 
G_M000_IG182:                ;; offset=0x162E
       E85D219AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG183:                ;; offset=0x1634
       488BBDA0FCFFFF       mov      rdi, gword ptr [rbp-0x360]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       76ED                 jbe      SHORT G_M000_IG182
       4C8BB5A0FCFFFF       mov      r14, gword ptr [rbp-0x360]
       4983C610             add      r14, 16
 
G_M000_IG184:                ;; offset=0x164C
       8BBD3CFEFFFF         mov      edi, dword ptr [rbp-0x1C4]
       8BB538FEFFFF         mov      esi, dword ptr [rbp-0x1C8]
       488B9528FEFFFF       mov      rdx, qword ptr [rbp-0x1D8]
       498BCE               mov      rcx, r14
       FF15786EB9FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8BBDBCFDFFFF         mov      edi, dword ptr [rbp-0x244]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4D8BC6               mov      r8, r14
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       FF15B7ECCBFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG185:                ;; offset=0x1692
       33FF                 xor      rdi, rdi
       4889BDB0FDFFFF       mov      gword ptr [rbp-0x250], rdi
 
G_M000_IG186:                ;; offset=0x169B
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB5A0FCFFFF       mov      rsi, gword ptr [rbp-0x360]
       33D2                 xor      edx, edx
       FF1581C0D5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       E904F5FFFF           jmp      G_M000_IG128
 
G_M000_IG187:                ;; offset=0x16BC
       4963F7               movsxd   rsi, r15d
       48C1E602             shl      rsi, 2
       498BFC               mov      rdi, r12
       49BBC0268455B67F0000 mov      r11, 0x7FB6558426C0
       41FF13               call     [r11]Lokad.Onnx.IScratchAccountant:AddScratchBytes(long):this
       E902F8FFFF           jmp      G_M000_IG147
 
G_M000_IG188:                ;; offset=0x16D8
       81BD3CFEFFFF000A0000 cmp      dword ptr [rbp-0x1C4], 0xA00
       7D3E                 jge      SHORT G_M000_IG190
 
G_M000_IG189:                ;; offset=0x16E4
       81BD38FEFFFF000A0000 cmp      dword ptr [rbp-0x1C8], 0xA00
       7D32                 jge      SHORT G_M000_IG190
       8BBDBCFDFFFF         mov      edi, dword ptr [rbp-0x244]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4C8B8528FEFFFF       mov      r8, qword ptr [rbp-0x1D8]
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       FF1503F6CBFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4tiled(int,int,int,ptr,ptr,ptr)]
       E99EF4FFFF           jmp      G_M000_IG128
 
G_M000_IG190:                ;; offset=0x1722
       8BBDBCFDFFFF         mov      edi, dword ptr [rbp-0x244]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4C8B8528FEFFFF       mov      r8, qword ptr [rbp-0x1D8]
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       FF15E9F5CBFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4(int,int,int,ptr,ptr,ptr)]
       E96CF4FFFF           jmp      G_M000_IG128
 
G_M000_IG191:                ;; offset=0x1754
       4C638D3CFEFFFF       movsxd   r9, dword ptr [rbp-0x1C4]
       48638D38FEFFFF       movsxd   rcx, dword ptr [rbp-0x1C8]
       4C0FAFC9             imul     r9, rcx
       4981F900000004       cmp      r9, 0x4000000
       0F8F78F4FFFF         jg       G_M000_IG129
       83BD40FEFFFF40       cmp      dword ptr [rbp-0x1C0], 64
       7D26                 jge      SHORT G_M000_IG192
       4C638D3CFEFFFF       movsxd   r9, dword ptr [rbp-0x1C4]
       48638D38FEFFFF       movsxd   rcx, dword ptr [rbp-0x1C8]
       4C0FAFC9             imul     r9, rcx
       4981F900000100       cmp      r9, 0x10000
       410F9EC1             setle    r9b
       450FB6C9             movzx    r9, r9b
       E94CF4FFFF           jmp      G_M000_IG130
 
G_M000_IG192:                ;; offset=0x17A2
       41B901000000         mov      r9d, 1
       E941F4FFFF           jmp      G_M000_IG130
 
G_M000_IG193:                ;; offset=0x17AD
       4585C9               test     r9d, r9d
       0F854AF4FFFF         jne      G_M000_IG131
       448B8DBCFDFFFF       mov      r9d, dword ptr [rbp-0x244]
       440FAF8D38FEFFFF     imul     r9d, dword ptr [rbp-0x1C8]
       4D63C9               movsxd   r9, r9d
       488B8D20FEFFFF       mov      rcx, qword ptr [rbp-0x1E0]
       4E8D0C89             lea      r9, [rcx+4*r9]
       8B8DBCFDFFFF         mov      ecx, dword ptr [rbp-0x244]
       0FAF8D3CFEFFFF       imul     ecx, dword ptr [rbp-0x1C4]
       4863C9               movsxd   rcx, ecx
       488BB530FEFFFF       mov      rsi, qword ptr [rbp-0x1D0]
       488D0C8E             lea      rcx, [rsi+4*rcx]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       4C8B8528FEFFFF       mov      r8, qword ptr [rbp-0x1D8]
       BF01000000           mov      edi, 1
       FF1544F5CBFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E9EFF3FFFF           jmp      G_M000_IG131
 
G_M000_IG194:                ;; offset=0x1811
       4585FF               test     r15d, r15d
       7432                 je       SHORT G_M000_IG195
       8BBD40FEFFFF         mov      edi, dword ptr [rbp-0x1C0]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4C8B8528FEFFFF       mov      r8, qword ptr [rbp-0x1D8]
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       FF150DF5CBFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E9B8F3FFFF           jmp      G_M000_IG131
 
G_M000_IG195:                ;; offset=0x1848
       8BBD40FEFFFF         mov      edi, dword ptr [rbp-0x1C0]
       8BB53CFEFFFF         mov      esi, dword ptr [rbp-0x1C4]
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       488B8D30FEFFFF       mov      rcx, qword ptr [rbp-0x1D0]
       4C8B8528FEFFFF       mov      r8, qword ptr [rbp-0x1D8]
       4C8B8D20FEFFFF       mov      r9, qword ptr [rbp-0x1E0]
       FF15F3F4CBFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized(int,int,int,ptr,ptr,ptr)]
       E986F3FFFF           jmp      G_M000_IG131
 
G_M000_IG196:                ;; offset=0x187A
       3BFB                 cmp      edi, ebx
       7378                 jae      SHORT G_M000_IG197
       33F6                 xor      esi, esi
       8974B910             mov      dword ptr [rcx+4*rdi+0x10], esi
       3B7808               cmp      edi, dword ptr [rax+0x08]
       736D                 jae      SHORT G_M000_IG197
       8B74B810             mov      esi, dword ptr [rax+4*rdi+0x10]
       3BFB                 cmp      edi, ebx
       7365                 jae      SHORT G_M000_IG197
       0FAF74BA10           imul     esi, dword ptr [rdx+4*rdi+0x10]
       448B9548FFFFFF       mov      r10d, dword ptr [rbp-0xB8]
       442BD6               sub      r10d, esi
       44899548FFFFFF       mov      dword ptr [rbp-0xB8], r10d
       413B7808             cmp      edi, dword ptr [r8+0x08]
       7349                 jae      SHORT G_M000_IG197
       418B74B810           mov      esi, dword ptr [r8+4*rdi+0x10]
       3BFB                 cmp      edi, ebx
       7340                 jae      SHORT G_M000_IG197
       0FAF74BA10           imul     esi, dword ptr [rdx+4*rdi+0x10]
       448B9544FFFFFF       mov      r10d, dword ptr [rbp-0xBC]
       442BD6               sub      r10d, esi
       44899544FFFFFF       mov      dword ptr [rbp-0xBC], r10d
       413B7908             cmp      edi, dword ptr [r9+0x08]
       7324                 jae      SHORT G_M000_IG197
       418B74B910           mov      esi, dword ptr [r9+4*rdi+0x10]
       3BFB                 cmp      edi, ebx
       731B                 jae      SHORT G_M000_IG197
       0FAF74BA10           imul     esi, dword ptr [rdx+4*rdi+0x10]
       448B9540FFFFFF       mov      r10d, dword ptr [rbp-0xC0]
       442BD6               sub      r10d, esi
       44899540FFFFFF       mov      dword ptr [rbp-0xC0], r10d
       E98DF8FFFF           jmp      G_M000_IG166
 
G_M000_IG197:                ;; offset=0x18F6
       E8951E9AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG198:                ;; offset=0x18FC
       488D7D88             lea      rdi, [rbp-0x78]
       FF15526CB9FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG199:                ;; offset=0x1907
       488D7DA0             lea      rdi, [rbp-0x60]
       FF15476CB9FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG200:                ;; offset=0x1912
       488D7DB8             lea      rdi, [rbp-0x48]
       FF153C6CB9FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       E95DF1FFFF           jmp      G_M000_IG116
 
G_M000_IG201:                ;; offset=0x1921
       4883F820             cmp      rax, 32
       736B                 jae      SHORT G_M000_IG209
 
G_M000_IG202:                ;; offset=0x1927
       4883F810             cmp      rax, 16
       7322                 jae      SHORT G_M000_IG205
 
G_M000_IG203:                ;; offset=0x192D
       4883C0F8             add      rax, -8
       488B11               mov      rdx, qword ptr [rcx]
       482B17               sub      rdx, qword ptr [rdi]
       488B3401             mov      rsi, qword ptr [rcx+rax]
       482B3407             sub      rsi, qword ptr [rdi+rax]
       480BD6               or       rdx, rsi
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
 
G_M000_IG204:                ;; offset=0x1948
       8BF8                 mov      edi, eax
       E952E9FFFF           jmp      G_M000_IG25
 
G_M000_IG205:                ;; offset=0x194F
       33C9                 xor      ecx, ecx
       4883C0F0             add      rax, -16
       7519                 jne      SHORT G_M000_IG207
 
G_M000_IG206:                ;; offset=0x1957
       C5F8100402           vmovups  xmm0, xmmword ptr [rdx+rax]
       62F17D08740C06       vpcmpeqb k1, xmm0, xmmword ptr [rsi+rax]
       C5F898C9             kortestw k1, k1
       736B                 jae      SHORT G_M000_IG213
       E92EE9FFFF           jmp      G_M000_IG24
 
G_M000_IG207:                ;; offset=0x196E
       C5F81002             vmovups  xmm0, xmmword ptr [rdx]
       62F37D083E0E04       vpcmpfalseub k1, xmm0, xmmword ptr [rsi]
       C5F898C9             kortestw k1, k1
       7553                 jne      SHORT G_M000_IG213
 
G_M000_IG208:                ;; offset=0x197F
       4883C110             add      rcx, 16
       483BC1               cmp      rax, rcx
       0F874F090000         ja       G_M000_IG308
       EBC9                 jmp      SHORT G_M000_IG206
 
G_M000_IG209:                ;; offset=0x198E
       33FF                 xor      edi, edi
       4883C0E0             add      rax, -32
       7422                 je       SHORT G_M000_IG212
 
G_M000_IG210:                ;; offset=0x1996
       C5FC1002             vmovups  ymm0, ymmword ptr [rdx]
       62F37D283E0E04       vpcmpfalseub k1, ymm0, ymmword ptr [rsi]
       C4E1F998C9           kortestd k1, k1
       7527                 jne      SHORT G_M000_IG213
 
G_M000_IG211:                ;; offset=0x19A8
       4883C720             add      rdi, 32
       483BC7               cmp      rax, rdi
       0F8703090000         ja       G_M000_IG307
 
G_M000_IG212:                ;; offset=0x19B5
       C5FC100402           vmovups  ymm0, ymmword ptr [rdx+rax]
       62F17D28740C06       vpcmpeqb k1, ymm0, ymmword ptr [rsi+rax]
       C4E1F998C9           kortestd k1, k1
       0F82D0E8FFFF         jb       G_M000_IG24
 
G_M000_IG213:                ;; offset=0x19CC
       33FF                 xor      edi, edi
       E9CEE8FFFF           jmp      G_M000_IG25
 
G_M000_IG214:                ;; offset=0x19D3
       4883F820             cmp      rax, 32
       736B                 jae      SHORT G_M000_IG222
 
G_M000_IG215:                ;; offset=0x19D9
       4883F810             cmp      rax, 16
       7322                 jae      SHORT G_M000_IG218
 
G_M000_IG216:                ;; offset=0x19DF
       4883C0F8             add      rax, -8
       488B11               mov      rdx, qword ptr [rcx]
       482B17               sub      rdx, qword ptr [rdi]
       488B3401             mov      rsi, qword ptr [rcx+rax]
       482B3407             sub      rsi, qword ptr [rdi+rax]
       480BD6               or       rdx, rsi
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
 
G_M000_IG217:                ;; offset=0x19FA
       8BF8                 mov      edi, eax
       E9F2E9FFFF           jmp      G_M000_IG41
 
G_M000_IG218:                ;; offset=0x1A01
       33C9                 xor      ecx, ecx
       4883C0F0             add      rax, -16
       7519                 jne      SHORT G_M000_IG220
 
G_M000_IG219:                ;; offset=0x1A09
       C5F8100402           vmovups  xmm0, xmmword ptr [rdx+rax]
       62F17D08740C06       vpcmpeqb k1, xmm0, xmmword ptr [rsi+rax]
       C5F898C9             kortestw k1, k1
       736B                 jae      SHORT G_M000_IG226
       E9CEE9FFFF           jmp      G_M000_IG40
 
G_M000_IG220:                ;; offset=0x1A20
       C5F81002             vmovups  xmm0, xmmword ptr [rdx]
       62F37D083E0E04       vpcmpfalseub k1, xmm0, xmmword ptr [rsi]
       C5F898C9             kortestw k1, k1
       7553                 jne      SHORT G_M000_IG226
 
G_M000_IG221:                ;; offset=0x1A31
       4883C110             add      rcx, 16
       483BC1               cmp      rax, rcx
       0F87510A0000         ja       G_M000_IG327
       EBC9                 jmp      SHORT G_M000_IG219
 
G_M000_IG222:                ;; offset=0x1A40
       33FF                 xor      edi, edi
       4883C0E0             add      rax, -32
       7422                 je       SHORT G_M000_IG225
 
G_M000_IG223:                ;; offset=0x1A48
       C5FC1002             vmovups  ymm0, ymmword ptr [rdx]
       62F37D283E0E04       vpcmpfalseub k1, ymm0, ymmword ptr [rsi]
       C4E1F998C9           kortestd k1, k1
       7527                 jne      SHORT G_M000_IG226
 
G_M000_IG224:                ;; offset=0x1A5A
       4883C720             add      rdi, 32
       483BC7               cmp      rax, rdi
       0F87050A0000         ja       G_M000_IG326
 
G_M000_IG225:                ;; offset=0x1A67
       C5FC100402           vmovups  ymm0, ymmword ptr [rdx+rax]
       62F17D28740C06       vpcmpeqb k1, ymm0, ymmword ptr [rsi+rax]
       C4E1F998C9           kortestd k1, k1
       0F8270E9FFFF         jb       G_M000_IG40
 
G_M000_IG226:                ;; offset=0x1A7E
       33FF                 xor      edi, edi
       E96EE9FFFF           jmp      G_M000_IG41
 
G_M000_IG227:                ;; offset=0x1A85
       4883F820             cmp      rax, 32
       736B                 jae      SHORT G_M000_IG235
 
G_M000_IG228:                ;; offset=0x1A8B
       4883F810             cmp      rax, 16
       7322                 jae      SHORT G_M000_IG231
 
G_M000_IG229:                ;; offset=0x1A91
       4883C0F8             add      rax, -8
       488B11               mov      rdx, qword ptr [rcx]
       482B17               sub      rdx, qword ptr [rdi]
       488B3401             mov      rsi, qword ptr [rcx+rax]
       482B3407             sub      rsi, qword ptr [rdi+rax]
       480BD6               or       rdx, rsi
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
 
G_M000_IG230:                ;; offset=0x1AAC
       8BF8                 mov      edi, eax
       E976EAFFFF           jmp      G_M000_IG56
 
G_M000_IG231:                ;; offset=0x1AB3
       33C9                 xor      ecx, ecx
       4883C0F0             add      rax, -16
       7519                 jne      SHORT G_M000_IG233
 
G_M000_IG232:                ;; offset=0x1ABB
       C5F8100402           vmovups  xmm0, xmmword ptr [rdx+rax]
       62F17D08740C06       vpcmpeqb k1, xmm0, xmmword ptr [rsi+rax]
       C5F898C9             kortestw k1, k1
       736B                 jae      SHORT G_M000_IG239
       E952EAFFFF           jmp      G_M000_IG55
 
G_M000_IG233:                ;; offset=0x1AD2
       C5F81002             vmovups  xmm0, xmmword ptr [rdx]
       62F37D083E0E04       vpcmpfalseub k1, xmm0, xmmword ptr [rsi]
       C5F898C9             kortestw k1, k1
       7553                 jne      SHORT G_M000_IG239
 
G_M000_IG234:                ;; offset=0x1AE3
       4883C110             add      rcx, 16
       483BC1               cmp      rax, rcx
       0F87260B0000         ja       G_M000_IG344
       EBC9                 jmp      SHORT G_M000_IG232
 
G_M000_IG235:                ;; offset=0x1AF2
       33FF                 xor      edi, edi
       4883C0E0             add      rax, -32
       7422                 je       SHORT G_M000_IG238
 
G_M000_IG236:                ;; offset=0x1AFA
       C5FC1002             vmovups  ymm0, ymmword ptr [rdx]
       62F37D283E0E04       vpcmpfalseub k1, ymm0, ymmword ptr [rsi]
       C4E1F998C9           kortestd k1, k1
       7527                 jne      SHORT G_M000_IG239
 
G_M000_IG237:                ;; offset=0x1B0C
       4883C720             add      rdi, 32
       483BC7               cmp      rax, rdi
       0F87DA0A0000         ja       G_M000_IG343
 
G_M000_IG238:                ;; offset=0x1B19
       C5FC100402           vmovups  ymm0, ymmword ptr [rdx+rax]
       62F17D28740C06       vpcmpeqb k1, ymm0, ymmword ptr [rsi+rax]
       C4E1F998C9           kortestd k1, k1
       0F82F4E9FFFF         jb       G_M000_IG55
 
G_M000_IG239:                ;; offset=0x1B30
       33FF                 xor      edi, edi
       E9F2E9FFFF           jmp      G_M000_IG56
 
G_M000_IG240:                ;; offset=0x1B37
       8BC6                 mov      eax, esi
       418B448210           mov      eax, dword ptr [r10+4*rax+0x10]
       E911ECFFFF           jmp      G_M000_IG75
 
G_M000_IG241:                ;; offset=0x1B43
       448B8D68FDFFFF       mov      r9d, dword ptr [rbp-0x298]
       458D41FE             lea      r8d, [r9-0x02]
       413BF0               cmp      esi, r8d
       7D1C                 jge      SHORT G_M000_IG243
 
G_M000_IG242:                ;; offset=0x1B53
       413BF1               cmp      esi, r9d
       0F83D60D0000         jae      G_M000_IG380
       8BC6                 mov      eax, esi
       488B9550FCFFFF       mov      rdx, bword ptr [rbp-0x3B0]
       833C8201             cmp      dword ptr [rdx+4*rax], 1
       0F85070C0000         jne      G_M000_IG360
 
G_M000_IG243:                ;; offset=0x1B6F
       33C0                 xor      eax, eax
 
G_M000_IG244:                ;; offset=0x1B71
       3B7108               cmp      esi, dword ptr [rcx+0x08]
       0F83B80D0000         jae      G_M000_IG380
       448BC6               mov      r8d, esi
       4289448110           mov      dword ptr [rcx+4*r8+0x10], eax
       FFC6                 inc      esi
       3BF7                 cmp      esi, edi
       7CBB                 jl       SHORT G_M000_IG241
       E9D5EBFFFF           jmp      G_M000_IG76
 
G_M000_IG245:                ;; offset=0x1B8D
       8BCE                 mov      ecx, esi
       418B4C8910           mov      ecx, dword ptr [r9+4*rcx+0x10]
       E9A4ECFFFF           jmp      G_M000_IG90
 
G_M000_IG246:                ;; offset=0x1B99
       448B9558FDFFFF       mov      r10d, dword ptr [rbp-0x2A8]
       418D52FE             lea      edx, [r10-0x02]
       3BF2                 cmp      esi, edx
       7D1D                 jge      SHORT G_M000_IG248
 
G_M000_IG247:                ;; offset=0x1BA8
       413BF2               cmp      esi, r10d
       0F83810D0000         jae      G_M000_IG380
       8BCE                 mov      ecx, esi
       4C8B8540FCFFFF       mov      r8, bword ptr [rbp-0x3C0]
       41833C8801           cmp      dword ptr [r8+4*rcx], 1
       0F85480C0000         jne      G_M000_IG367
 
G_M000_IG248:                ;; offset=0x1BC5
       33C9                 xor      ecx, ecx
 
G_M000_IG249:                ;; offset=0x1BC7
       3B7708               cmp      esi, dword ptr [rdi+0x08]
       0F83620D0000         jae      G_M000_IG380
       8BD6                 mov      edx, esi
       894C9710             mov      dword ptr [rdi+4*rdx+0x10], ecx
       FFC6                 inc      esi
       3BF0                 cmp      esi, eax
       7CBD                 jl       SHORT G_M000_IG246
       E96AECFFFF           jmp      G_M000_IG91
 
G_M000_IG250:                ;; offset=0x1BE1
       8BC8                 mov      ecx, eax
       8B4C8F10             mov      ecx, dword ptr [rdi+4*rcx+0x10]
       E909EDFFFF           jmp      G_M000_IG101
 
G_M000_IG251:                ;; offset=0x1BEC
       448B9550FDFFFF       mov      r10d, dword ptr [rbp-0x2B0]
       418D72FE             lea      esi, [r10-0x02]
       3BC6                 cmp      eax, esi
       7D1D                 jge      SHORT G_M000_IG253
 
G_M000_IG252:                ;; offset=0x1BFB
       413BC2               cmp      eax, r10d
       0F832E0D0000         jae      G_M000_IG380
       8BC8                 mov      ecx, eax
       4C8B8D38FCFFFF       mov      r9, bword ptr [rbp-0x3C8]
       41833C8901           cmp      dword ptr [r9+4*rcx], 1
       0F85230C0000         jne      G_M000_IG372
 
G_M000_IG253:                ;; offset=0x1C18
       33C9                 xor      ecx, ecx
 
G_M000_IG254:                ;; offset=0x1C1A
       413B4008             cmp      eax, dword ptr [r8+0x08]
       0F830E0D0000         jae      G_M000_IG380
       8BF0                 mov      esi, eax
       41894CB010           mov      dword ptr [r8+4*rsi+0x10], ecx
       FFC0                 inc      eax
       3BC2                 cmp      eax, edx
       7CBB                 jl       SHORT G_M000_IG251
       E9CDECFFFF           jmp      G_M000_IG102
 
G_M000_IG255:                ;; offset=0x1C36
       33D2                 xor      rdx, rdx
       33C9                 xor      ecx, ecx
       E97CE4FFFF           jmp      G_M000_IG05
 
G_M000_IG256:                ;; offset=0x1C3F
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E994E4FFFF           jmp      G_M000_IG06
 
G_M000_IG257:                ;; offset=0x1C48
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E9B0E4FFFF           jmp      G_M000_IG07
 
G_M000_IG258:                ;; offset=0x1C51
       4181FC00040000       cmp      r12d, 0x400
       0F8CE5E4FFFF         jl       G_M000_IG09
       81F900040000         cmp      ecx, 0x400
       0F8CD9E4FFFF         jl       G_M000_IG09
       418BFC               mov      edi, r12d
       8BD1                 mov      edx, ecx
       480FAFFA             imul     rdi, rdx
       4881FF00000004       cmp      rdi, 0x4000000
       7E0B                 jle      SHORT G_M000_IG259
       8B8D0CFFFFFF         mov      ecx, dword ptr [rbp-0xF4]
       E9BCE4FFFF           jmp      G_M000_IG09
 
G_M000_IG259:                ;; offset=0x1C87
       488BFB               mov      rdi, rbx
       FF15C0CFCBFF         call     [Lokad.Onnx.OwnedPackedTensor:Resolve(Lokad.Onnx.Tensor`1[float]):Lokad.Onnx.OwnedPackedTensor]
       4885C0               test     rax, rax
       0F849E000000         je       G_M000_IG263
       44396048             cmp      dword ptr [rax+0x48], r12d
       0F8589000000         jne      G_M000_IG262
       8B8D0CFFFFFF         mov      ecx, dword ptr [rbp-0xF4]
       39484C               cmp      dword ptr [rax+0x4C], ecx
       7573                 jne      SHORT G_M000_IG261
       488BD0               mov      rdx, rax
       8B8D0CFFFFFF         mov      ecx, dword ptr [rbp-0xF4]
       E98BE4FFFF           jmp      G_M000_IG10
 
G_M000_IG260:                ;; offset=0x1CBC
       488DBDD8FEFFFF       lea      rdi, [rbp-0x128]
       FF1557C3B9FF         call     [Lokad.Onnx.TensorExecutionOptions:Validate():this]
       498BFF               mov      rdi, r15
       488B95E0FEFFFF       mov      rdx, gword ptr [rbp-0x120]
       48BE18F6C045B67F0000 mov      rsi, 0x7FB645C0F618
       FF156DE4CBFF         call     [Lokad.Onnx.Tensor`1[float]:RequireBatchOperand(Lokad.Onnx.Tensor`1[float],System.String,Lokad.Onnx.ICopyAccountant):Lokad.Onnx.Tensor`1[float]]
       48898510FDFFFF       mov      gword ptr [rbp-0x2F0], rax
       498BFE               mov      rdi, r14
       488B95E0FEFFFF       mov      rdx, gword ptr [rbp-0x120]
       48BE58F6C045B67F0000 mov      rsi, 0x7FB645C0F658
       FF151CCFCBFF         call     [Lokad.Onnx.Tensor`1[float]:RequireContiguous[float](Lokad.Onnx.Tensor`1[float],System.String,Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       48898508FDFFFF       mov      gword ptr [rbp-0x2F8], rax
       488B8D10FDFFFF       mov      rcx, gword ptr [rbp-0x2F0]
       488B7910             mov      rdi, gword ptr [rcx+0x10]
       4885FF               test     rdi, rdi
       7527                 jne      SHORT G_M000_IG264
       33D2                 xor      rdx, rdx
       33FF                 xor      edi, edi
       EB28                 jmp      SHORT G_M000_IG265
 
G_M000_IG261:                ;; offset=0x1D21
       8B8D0CFFFFFF         mov      ecx, dword ptr [rbp-0xF4]
       E917E4FFFF           jmp      G_M000_IG09
 
G_M000_IG262:                ;; offset=0x1D2C
       8B8D0CFFFFFF         mov      ecx, dword ptr [rbp-0xF4]
       E90CE4FFFF           jmp      G_M000_IG09
 
G_M000_IG263:                ;; offset=0x1D37
       8B8D0CFFFFFF         mov      ecx, dword ptr [rbp-0xF4]
       E901E4FFFF           jmp      G_M000_IG09
 
G_M000_IG264:                ;; offset=0x1D42
       488D5710             lea      rdx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG265:                ;; offset=0x1D49
       48899510FFFFFF       mov      bword ptr [rbp-0xF0], rdx
       89BD18FFFFFF         mov      dword ptr [rbp-0xE8], edi
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       83C7FE               add      edi, -2
       3BBD18FFFFFF         cmp      edi, dword ptr [rbp-0xE8]
       0F8754090000         ja       G_M000_IG349
       488B9510FFFFFF       mov      rdx, bword ptr [rbp-0xF0]
       488995C8FEFFFF       mov      bword ptr [rbp-0x138], rdx
       89BDD0FEFFFF         mov      dword ptr [rbp-0x130], edi
       488DBDC8FEFFFF       lea      rdi, [rbp-0x138]
       FF158C53ADFF         call     [System.ReadOnlySpan`1[int]:ToArray():int[]:this]
       48898500FDFFFF       mov      gword ptr [rbp-0x300], rax
       4885C0               test     rax, rax
       7513                 jne      SHORT G_M000_IG266
       33C9                 xor      rcx, rcx
       48898D80FCFFFF       mov      bword ptr [rbp-0x380], rcx
       33D2                 xor      edx, edx
       899588FDFFFF         mov      dword ptr [rbp-0x278], edx
       EB1B                 jmp      SHORT G_M000_IG267
 
G_M000_IG266:                ;; offset=0x1DAB
       488D4810             lea      rcx, bword ptr [rax+0x10]
       8B5008               mov      edx, dword ptr [rax+0x08]
       48898D80FCFFFF       mov      bword ptr [rbp-0x380], rcx
       899588FDFFFF         mov      dword ptr [rbp-0x278], edx
       488B8500FDFFFF       mov      rax, gword ptr [rbp-0x300]
 
G_M000_IG267:                ;; offset=0x1DC6
       488BB510FDFFFF       mov      rsi, gword ptr [rbp-0x2F0]
       488B7E10             mov      rdi, gword ptr [rsi+0x10]
       4885FF               test     rdi, rdi
       7516                 jne      SHORT G_M000_IG268
       4533C0               xor      r8, r8
       4C898578FCFFFF       mov      bword ptr [rbp-0x388], r8
       4533C9               xor      r9d, r9d
       44898D84FDFFFF       mov      dword ptr [rbp-0x27C], r9d
       EB16                 jmp      SHORT G_M000_IG269
 
G_M000_IG268:                ;; offset=0x1DEC
       4C8D4710             lea      r8, bword ptr [rdi+0x10]
       448B4F08             mov      r9d, dword ptr [rdi+0x08]
       4C898578FCFFFF       mov      bword ptr [rbp-0x388], r8
       44898D84FDFFFF       mov      dword ptr [rbp-0x27C], r9d
 
G_M000_IG269:                ;; offset=0x1E02
       488BFE               mov      rdi, rsi
       FF155DE3CBFF         call     [Lokad.Onnx.Tensor`1[float]:BatchStrides(Lokad.Onnx.Tensor`1[float]):int[]]
       4C8BC0               mov      r8, rax
       488B9578FCFFFF       mov      rdx, bword ptr [rbp-0x388]
       8B8D84FDFFFF         mov      ecx, dword ptr [rbp-0x27C]
       488BBD80FCFFFF       mov      rdi, bword ptr [rbp-0x380]
       8BB588FDFFFF         mov      esi, dword ptr [rbp-0x278]
       FF1552E3CBFF         call     [Lokad.Onnx.Tensor`1[float]:BatchSteps(System.ReadOnlySpan`1[int],System.ReadOnlySpan`1[int],int[]):int[]]
       488985F8FCFFFF       mov      gword ptr [rbp-0x308], rax
       4C8B8D00FDFFFF       mov      r9, gword ptr [rbp-0x300]
       4D85C9               test     r9, r9
       7506                 jne      SHORT G_M000_IG270
       33FF                 xor      rdi, rdi
       33F6                 xor      esi, esi
       EB0F                 jmp      SHORT G_M000_IG271
 
G_M000_IG270:                ;; offset=0x1E47
       498D7910             lea      rdi, bword ptr [r9+0x10]
       418B7108             mov      esi, dword ptr [r9+0x08]
       4C8B8D00FDFFFF       mov      r9, gword ptr [rbp-0x300]
 
G_M000_IG271:                ;; offset=0x1E56
       4C8B9508FDFFFF       mov      r10, gword ptr [rbp-0x2F8]
       498B5210             mov      rdx, gword ptr [r10+0x10]
       4885D2               test     rdx, rdx
       7507                 jne      SHORT G_M000_IG272
       33C9                 xor      rcx, rcx
       4533C0               xor      r8d, r8d
       EB08                 jmp      SHORT G_M000_IG273
 
G_M000_IG272:                ;; offset=0x1E6D
       488D4A10             lea      rcx, bword ptr [rdx+0x10]
       448B4208             mov      r8d, dword ptr [rdx+0x08]
 
G_M000_IG273:                ;; offset=0x1E75
       488BD1               mov      rdx, rcx
       418BC8               mov      ecx, r8d
       4D8B4218             mov      r8, gword ptr [r10+0x18]
       FF15FBE2CBFF         call     [Lokad.Onnx.Tensor`1[float]:BatchSteps(System.ReadOnlySpan`1[int],System.ReadOnlySpan`1[int],int[]):int[]]
       488985F0FCFFFF       mov      gword ptr [rbp-0x310], rax
       488B8D00FDFFFF       mov      rcx, gword ptr [rbp-0x300]
       4885C9               test     rcx, rcx
       7506                 jne      SHORT G_M000_IG274
       33FF                 xor      rdi, rdi
       33F6                 xor      esi, esi
       EB07                 jmp      SHORT G_M000_IG275
 
G_M000_IG274:                ;; offset=0x1E9E
       488D7910             lea      rdi, bword ptr [rcx+0x10]
       8B7108               mov      esi, dword ptr [rcx+0x08]
 
G_M000_IG275:                ;; offset=0x1EA5
       FF15EDE2CBFF         call     [Lokad.Onnx.Tensor`1[float]:BatchCount(System.ReadOnlySpan`1[int]):int]
       8985C4FEFFFF         mov      dword ptr [rbp-0x13C], eax
       85C0                 test     eax, eax
       750A                 jne      SHORT G_M000_IG276
       BE01000000           mov      esi, 1
       E99AE2FFFF           jmp      G_M000_IG11
 
G_M000_IG276:                ;; offset=0x1EBF
       488BBD10FDFFFF       mov      rdi, gword ptr [rbp-0x2F0]
       FF15B4DFCBFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       488985B0FEFFFF       mov      gword ptr [rbp-0x150], rax
       488995B8FEFFFF       mov      qword ptr [rbp-0x148], rdx
       488DBDB0FEFFFF       lea      rdi, [rbp-0x150]
       488DB598FEFFFF       lea      rsi, [rbp-0x168]
       FF15C265B9FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG277:                ;; offset=0x1EEF
       488BBD08FDFFFF       mov      rdi, gword ptr [rbp-0x2F8]
       FF1584DFCBFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       488985B0FEFFFF       mov      gword ptr [rbp-0x150], rax
       488995B8FEFFFF       mov      qword ptr [rbp-0x148], rdx
       488DBDB0FEFFFF       lea      rdi, [rbp-0x150]
       488DB580FEFFFF       lea      rsi, [rbp-0x180]
       FF159265B9FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG278:                ;; offset=0x1F1F
       FF15E3CCB9FF         call     [Lokad.Onnx.Profiler:get_Current():Lokad.Onnx.ProfilerContext]
       488985D8FCFFFF       mov      gword ptr [rbp-0x328], rax
       80783400             cmp      byte  ptr [rax+0x34], 0
       0F848A000000         je       G_M000_IG281
       80783600             cmp      byte  ptr [rax+0x36], 0
       0F8580000000         jne      G_M000_IG281
       488B7818             mov      rdi, gword ptr [rax+0x18]
       4889BDD0FCFFFF       mov      gword ptr [rbp-0x330], rdi
       C68548FEFFFF00       mov      byte  ptr [rbp-0x1B8], 0
 
G_M000_IG279:                ;; offset=0x1F52
       488DB548FEFFFF       lea      rsi, [rbp-0x1B8]
       488BBDD0FCFFFF       mov      rdi, gword ptr [rbp-0x330]
       FF153A5A9AFE         call     [System.Threading.Monitor:Enter(System.Object,byref)]
       488BBDD8FCFFFF       mov      rdi, gword ptr [rbp-0x328]
       FF156DD0B9FF         call     [Lokad.Onnx.ProfilerContext:AddTimeLocked():this]
       488B85D8FCFFFF       mov      rax, gword ptr [rbp-0x328]
       488B7808             mov      rdi, gword ptr [rax+0x08]
       393F                 cmp      dword ptr [rdi], edi
       FF15825ECBFF         call     [System.Collections.Generic.Stack`1[System.__Canon]:Peek():System.__Canon:this]
       488B7810             mov      rdi, gword ptr [rax+0x10]
       33F6                 xor      esi, esi
       33D2                 xor      edx, edx
       393F                 cmp      dword ptr [rdi], edi
       FF15AAD0B9FF         call     [System.Collections.Generic.Stack`1[Lokad.Onnx.OpProfile]:Push(Lokad.Onnx.OpProfile):this]
       488BBDD8FCFFFF       mov      rdi, gword ptr [rbp-0x328]
       488B7F10             mov      rdi, gword ptr [rdi+0x10]
       393F                 cmp      dword ptr [rdi], edi
       FF15AFD0B9FF         call     [System.Diagnostics.Stopwatch:Start():this]
       90                   nop      
 
G_M000_IG280:                ;; offset=0x1FAA
       80BD48FEFFFF00       cmp      byte  ptr [rbp-0x1B8], 0
       740D                 je       SHORT G_M000_IG281
       488BBDD0FCFFFF       mov      rdi, gword ptr [rbp-0x330]
       FF1568689AFE         call     [System.Threading.Monitor:Exit(System.Object)]
 
G_M000_IG281:                ;; offset=0x1FC0
       488BB5E0FCFFFF       mov      rsi, gword ptr [rbp-0x320]
       488B4640             mov      rax, gword ptr [rsi+0x40]
       48898578FEFFFF       mov      gword ptr [rbp-0x188], rax
       4885C0               test     rax, rax
       7406                 je       SHORT G_M000_IG282
       83780800             cmp      dword ptr [rax+0x08], 0
       750B                 jne      SHORT G_M000_IG283
 
G_M000_IG282:                ;; offset=0x1FDD
       33C0                 xor      eax, eax
       48898570FEFFFF       mov      qword ptr [rbp-0x190], rax
       EB0B                 jmp      SHORT G_M000_IG284
 
G_M000_IG283:                ;; offset=0x1FE8
       4883C010             add      rax, 16
       48898570FEFFFF       mov      qword ptr [rbp-0x190], rax
 
G_M000_IG284:                ;; offset=0x1FF3
       488B8DA0FEFFFF       mov      rcx, qword ptr [rbp-0x160]
       48898D68FEFFFF       mov      qword ptr [rbp-0x198], rcx
       488B9588FEFFFF       mov      rdx, qword ptr [rbp-0x178]
       48899560FEFFFF       mov      qword ptr [rbp-0x1A0], rdx
       4533C0               xor      r8d, r8d
       4489855CFEFFFF       mov      dword ptr [rbp-0x1A4], r8d
       4533C9               xor      r9d, r9d
       44898D58FEFFFF       mov      dword ptr [rbp-0x1A8], r9d
       4C8B9500FDFFFF       mov      r10, gword ptr [rbp-0x300]
       418B7208             mov      esi, dword ptr [r10+0x08]
       48BF58B18556B67F0000 mov      rdi, 0x7FB65685B158
       E833AAF77C           call     CORINFO_HELP_NEWARR_1_VC
       488985E8FCFFFF       mov      gword ptr [rbp-0x318], rax
       4533D2               xor      r10d, r10d
       E91B010000           jmp      G_M000_IG290
 
G_M000_IG285:                ;; offset=0x204C
       4C638D58FEFFFF       movsxd   r9, dword ptr [rbp-0x1A8]
       488B9560FEFFFF       mov      rdx, qword ptr [rbp-0x1A0]
       4E8D0C8A             lea      r9, [rdx+4*r9]
       48638D5CFEFFFF       movsxd   rcx, dword ptr [rbp-0x1A4]
       488BBD68FEFFFF       mov      rdi, qword ptr [rbp-0x198]
       488D0C8F             lea      rcx, [rdi+4*rcx]
       418BFD               mov      edi, r13d
       418BF4               mov      esi, r12d
       8B950CFFFFFF         mov      edx, dword ptr [rbp-0xF4]
       4C8B8570FEFFFF       mov      r8, qword ptr [rbp-0x190]
       FF15E7E1CBFF         call     [Lokad.Onnx.Tensor`1[float]:RunOwnedPackedRows(int,int,int,ptr,ptr,ptr)]
       488BBD00FDFFFF       mov      rdi, gword ptr [rbp-0x300]
       8B4708               mov      eax, dword ptr [rdi+0x08]
       FFC8                 dec      eax
       E98C000000           jmp      G_M000_IG288
 
G_M000_IG286:                ;; offset=0x209A
       E8F1169AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG287:                ;; offset=0x20A0
       488B8DE8FCFFFF       mov      rcx, gword ptr [rbp-0x318]
       3B4108               cmp      eax, dword ptr [rcx+0x08]
       73EE                 jae      SHORT G_M000_IG286
       488D548110           lea      rdx, bword ptr [rcx+4*rax+0x10]
       FF02                 inc      dword ptr [rdx]
       488B95F8FCFFFF       mov      rdx, gword ptr [rbp-0x308]
       3B4208               cmp      eax, dword ptr [rdx+0x08]
       73DB                 jae      SHORT G_M000_IG286
       8BB55CFEFFFF         mov      esi, dword ptr [rbp-0x1A4]
       03748210             add      esi, dword ptr [rdx+4*rax+0x10]
       4C8B85F0FCFFFF       mov      r8, gword ptr [rbp-0x310]
       413B4008             cmp      eax, dword ptr [r8+0x08]
       73C4                 jae      SHORT G_M000_IG286
       448B8D58FEFFFF       mov      r9d, dword ptr [rbp-0x1A8]
       45034C8010           add      r9d, dword ptr [r8+4*rax+0x10]
       448B548110           mov      r10d, dword ptr [rcx+4*rax+0x10]
       3B4708               cmp      eax, dword ptr [rdi+0x08]
       73AE                 jae      SHORT G_M000_IG286
       443B548710           cmp      r10d, dword ptr [rdi+4*rax+0x10]
       7C5D                 jl       SHORT G_M000_IG289
       4533D2               xor      r10d, r10d
       4489548110           mov      dword ptr [rcx+4*rax+0x10], r10d
       448B548210           mov      r10d, dword ptr [rdx+4*rax+0x10]
       440FAF548710         imul     r10d, dword ptr [rdi+4*rax+0x10]
       412BF2               sub      esi, r10d
       458B548010           mov      r10d, dword ptr [r8+4*rax+0x10]
       440FAF548710         imul     r10d, dword ptr [rdi+4*rax+0x10]
       452BCA               sub      r9d, r10d
       FFC8                 dec      eax
       89B55CFEFFFF         mov      dword ptr [rbp-0x1A4], esi
       44898D58FEFFFF       mov      dword ptr [rbp-0x1A8], r9d
 
G_M000_IG288:                ;; offset=0x2126
       85C0                 test     eax, eax
       0F8D72FFFFFF         jge      G_M000_IG287
       488B8DE8FCFFFF       mov      rcx, gword ptr [rbp-0x318]
       488B95F8FCFFFF       mov      rdx, gword ptr [rbp-0x308]
       8BB55CFEFFFF         mov      esi, dword ptr [rbp-0x1A4]
       4C8B85F0FCFFFF       mov      r8, gword ptr [rbp-0x310]
       448B8D58FEFFFF       mov      r9d, dword ptr [rbp-0x1A8]
 
G_M000_IG289:                ;; offset=0x2150
       448B9554FEFFFF       mov      r10d, dword ptr [rbp-0x1AC]
       41FFC2               inc      r10d
       89B55CFEFFFF         mov      dword ptr [rbp-0x1A4], esi
       44898D58FEFFFF       mov      dword ptr [rbp-0x1A8], r9d
 
G_M000_IG290:                ;; offset=0x2167
       44899554FEFFFF       mov      dword ptr [rbp-0x1AC], r10d
       443B95C4FEFFFF       cmp      r10d, dword ptr [rbp-0x13C]
       0F8CD1FEFFFF         jl       G_M000_IG285
       33FF                 xor      rdi, rdi
       4889BD78FEFFFF       mov      gword ptr [rbp-0x188], rdi
 
G_M000_IG291:                ;; offset=0x2184
       488DBD80FEFFFF       lea      rdi, [rbp-0x180]
       FF15C763B9FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG292:                ;; offset=0x2192
       488DBD98FEFFFF       lea      rdi, [rbp-0x168]
       FF15B963B9FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       BE01000000           mov      esi, 1
       E9B0DFFFFF           jmp      G_M000_IG11
 
G_M000_IG293:                ;; offset=0x21A9
       498BF7               mov      rsi, r15
       48BF3834BA57B67F0000 mov      rdi, 0x7FB657BA3438
       E8C55FC6FF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       488BF8               mov      rdi, rax
       E9CADFFFFF           jmp      G_M000_IG12
 
G_M000_IG294:                ;; offset=0x21C3
       FF15BFE0CBFF         call     [Lokad.Onnx.Tensor`1[float]:HasDenseMatrixCore(Lokad.Onnx.BroadcastedTensor`1[float]):bool]
       85C0                 test     eax, eax
       0F84C5DFFFFF         je       G_M000_IG13
       E9E6E0FFFF           jmp      G_M000_IG27
 
G_M000_IG295:                ;; offset=0x21D6
       498BF7               mov      rsi, r15
       E8A25FC6FF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BE0               mov      r12, rax
       E9C7DFFFFF           jmp      G_M000_IG14
 
G_M000_IG296:                ;; offset=0x21E6
       33C9                 xor      rcx, rcx
       48898D70FCFFFF       mov      bword ptr [rbp-0x390], rcx
       4533C0               xor      r8d, r8d
       44898580FDFFFF       mov      dword ptr [rbp-0x280], r8d
       488B8D70FCFFFF       mov      rcx, bword ptr [rbp-0x390]
       448B8580FDFFFF       mov      r8d, dword ptr [rbp-0x280]
       E9CCDFFFFF           jmp      G_M000_IG15
 
G_M000_IG297:                ;; offset=0x220C
       33F6                 xor      rsi, rsi
       33D2                 xor      edx, edx
       48898D70FCFFFF       mov      bword ptr [rbp-0x390], rcx
       44898580FDFFFF       mov      dword ptr [rbp-0x280], r8d
       E9D8DFFFFF           jmp      G_M000_IG16
 
G_M000_IG298:                ;; offset=0x2223
       33FF                 xor      rdi, rdi
       33C0                 xor      eax, eax
       E9ECDFFFFF           jmp      G_M000_IG17
 
G_M000_IG299:                ;; offset=0x222C
       4883F804             cmp      rax, 4
       7332                 jae      SHORT G_M000_IG305
 
G_M000_IG300:                ;; offset=0x2232
       33D2                 xor      edx, edx
       488BF0               mov      rsi, rax
       4883E602             and      rsi, 2
       740A                 je       SHORT G_M000_IG302
 
G_M000_IG301:                ;; offset=0x223D
       0FB711               movzx    rdx, word  ptr [rcx]
       440FB707             movzx    r8, word  ptr [rdi]
       412BD0               sub      edx, r8d
 
G_M000_IG302:                ;; offset=0x2247
       A801                 test     al, 1
       740C                 je       SHORT G_M000_IG304
 
G_M000_IG303:                ;; offset=0x224B
       0FB60431             movzx    rax, byte  ptr [rcx+rsi]
       0FB60C37             movzx    rcx, byte  ptr [rdi+rsi]
       2BC1                 sub      eax, ecx
       0BD0                 or       edx, eax
 
G_M000_IG304:                ;; offset=0x2257
       85D2                 test     edx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E9E4F6FFFF           jmp      G_M000_IG204
 
G_M000_IG305:                ;; offset=0x2264
       4883C0FC             add      rax, -4
       8B11                 mov      edx, dword ptr [rcx]
       2B17                 sub      edx, dword ptr [rdi]
       8B0C01               mov      ecx, dword ptr [rcx+rax]
       2B0C07               sub      ecx, dword ptr [rdi+rax]
       0BCA                 or       ecx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E9C9F6FFFF           jmp      G_M000_IG204
 
G_M000_IG306:                ;; offset=0x227F
       62F17C4810040A       vmovups  zmm0, zmmword ptr [rdx+rcx]
       62F37D483E0C0E04     vpcmpfalseub k1, zmm0, zmmword ptr [rsi+rcx]
       C4E1F898C9           kortestq k1, k1
       0F8533F7FFFF         jne      G_M000_IG213
       E9D5DFFFFF           jmp      G_M000_IG22
 
G_M000_IG307:                ;; offset=0x229E
       C5FC10043A           vmovups  ymm0, ymmword ptr [rdx+rdi]
       62F37D283E0C3E04     vpcmpfalseub k1, ymm0, ymmword ptr [rsi+rdi]
       C4E1F998C9           kortestd k1, k1
       0F8516F7FFFF         jne      G_M000_IG213
       E9EDF6FFFF           jmp      G_M000_IG211
 
G_M000_IG308:                ;; offset=0x22BB
       C5F810040A           vmovups  xmm0, xmmword ptr [rdx+rcx]
       62F37D083E0C0E04     vpcmpfalseub k1, xmm0, xmmword ptr [rsi+rcx]
       C5F898C9             kortestw k1, k1
       0F85FAF6FFFF         jne      G_M000_IG213
       E9A8F6FFFF           jmp      G_M000_IG208
 
G_M000_IG309:                ;; offset=0x22D7
       33FF                 xor      edi, edi
       E9C3DFFFFF           jmp      G_M000_IG25
 
G_M000_IG310:                ;; offset=0x22DE
       48BF28CD8656B67F0000 mov      rdi, 0x7FB65686CD28
       E873A6F77C           call     CORINFO_HELP_NEWSFAST
       4C8BF8               mov      r15, rax
       BFCBE70000           mov      edi, 0xE7CB
       48BE087F9256B67F0000 mov      rsi, 0x7FB656927F08
       FF15234161FF         call     [CORINFO_HELP_STRCNS]
       4C8BE8               mov      r13, rax
       BF97EA0000           mov      edi, 0xEA97
       48BE087F9256B67F0000 mov      rsi, 0x7FB656927F08
       FF150B4161FF         call     [CORINFO_HELP_STRCNS]
       488BF0               mov      rsi, rax
       498BFD               mov      rdi, r13
       FF155FD99AFE         call     [System.String:Concat(System.String,System.String):System.String]
       488BF0               mov      rsi, rax
       498BFF               mov      rdi, r15
       FF15534161FF         call     [System.ArgumentException:.ctor(System.String):this]
       498BFF               mov      rdi, r15
       E8C3D2E27C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG311:                ;; offset=0x233E
       498BFF               mov      rdi, r15
       498B0F               mov      rcx, qword ptr [r15]
       488B4978             mov      rcx, qword ptr [rcx+0x78]
       FF5110               call     [rcx+0x10]Lokad.Onnx.Tensor`1[float]:ToDenseTensor():Lokad.Onnx.DenseTensor`1[float]:this
       488BF8               mov      rdi, rax
       498BF5               mov      rsi, r13
       FF1579CDCBFF         call     [Lokad.Onnx.Tensor`1[float]:CountedCopy[float](Lokad.Onnx.DenseTensor`1[float],Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE0               mov      r12, rax
       E95ADFFFFF           jmp      G_M000_IG26
 
G_M000_IG312:                ;; offset=0x235F
       488BF3               mov      rsi, rbx
       48BF3834BA57B67F0000 mov      rdi, 0x7FB657BA3438
       E80F5EC6FF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       488BF8               mov      rdi, rax
       E966DFFFFF           jmp      G_M000_IG28
 
G_M000_IG313:                ;; offset=0x2379
       FF1509DFCBFF         call     [Lokad.Onnx.Tensor`1[float]:HasDenseMatrixCore(Lokad.Onnx.BroadcastedTensor`1[float]):bool]
       85C0                 test     eax, eax
       0F8461DFFFFF         je       G_M000_IG29
       E982E0FFFF           jmp      G_M000_IG43
 
G_M000_IG314:                ;; offset=0x238C
       488BF3               mov      rsi, rbx
       E8EC5DC6FF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BE0               mov      r12, rax
       E963DFFFFF           jmp      G_M000_IG30
 
G_M000_IG315:                ;; offset=0x239C
       33C9                 xor      rcx, rcx
       48898D68FCFFFF       mov      bword ptr [rbp-0x398], rcx
       4533C0               xor      r8d, r8d
       4489857CFDFFFF       mov      dword ptr [rbp-0x284], r8d
       488B8D68FCFFFF       mov      rcx, bword ptr [rbp-0x398]
       448B857CFDFFFF       mov      r8d, dword ptr [rbp-0x284]
       E968DFFFFF           jmp      G_M000_IG31
 
G_M000_IG316:                ;; offset=0x23C2
       33F6                 xor      rsi, rsi
       33D2                 xor      edx, edx
       48898D68FCFFFF       mov      bword ptr [rbp-0x398], rcx
       4489857CFDFFFF       mov      dword ptr [rbp-0x284], r8d
       E974DFFFFF           jmp      G_M000_IG32
 
G_M000_IG317:                ;; offset=0x23D9
       33FF                 xor      rdi, rdi
       33C0                 xor      eax, eax
       E988DFFFFF           jmp      G_M000_IG33
 
G_M000_IG318:                ;; offset=0x23E2
       4883F804             cmp      rax, 4
       7332                 jae      SHORT G_M000_IG324
 
G_M000_IG319:                ;; offset=0x23E8
       33D2                 xor      edx, edx
       488BF0               mov      rsi, rax
       4883E602             and      rsi, 2
       740A                 je       SHORT G_M000_IG321
 
G_M000_IG320:                ;; offset=0x23F3
       0FB711               movzx    rdx, word  ptr [rcx]
       440FB707             movzx    r8, word  ptr [rdi]
       412BD0               sub      edx, r8d
 
G_M000_IG321:                ;; offset=0x23FD
       A801                 test     al, 1
       740C                 je       SHORT G_M000_IG323
 
G_M000_IG322:                ;; offset=0x2401
       0FB60431             movzx    rax, byte  ptr [rcx+rsi]
       0FB60C37             movzx    rcx, byte  ptr [rdi+rsi]
       2BC1                 sub      eax, ecx
       0BD0                 or       edx, eax
 
G_M000_IG323:                ;; offset=0x240D
       85D2                 test     edx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E9E0F5FFFF           jmp      G_M000_IG217
 
G_M000_IG324:                ;; offset=0x241A
       4883C0FC             add      rax, -4
       8B11                 mov      edx, dword ptr [rcx]
       2B17                 sub      edx, dword ptr [rdi]
       8B0C01               mov      ecx, dword ptr [rcx+rax]
       2B0C07               sub      ecx, dword ptr [rdi+rax]
       0BCA                 or       ecx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E9C5F5FFFF           jmp      G_M000_IG217
 
G_M000_IG325:                ;; offset=0x2435
       62F17C4810040A       vmovups  zmm0, zmmword ptr [rdx+rcx]
       62F37D483E0C0E04     vpcmpfalseub k1, zmm0, zmmword ptr [rsi+rcx]
       C4E1F898C9           kortestq k1, k1
       0F852FF6FFFF         jne      G_M000_IG226
       E971DFFFFF           jmp      G_M000_IG38
 
G_M000_IG326:                ;; offset=0x2454
       C5FC10043A           vmovups  ymm0, ymmword ptr [rdx+rdi]
       62F37D283E0C3E04     vpcmpfalseub k1, ymm0, ymmword ptr [rsi+rdi]
       C4E1F998C9           kortestd k1, k1
       0F8512F6FFFF         jne      G_M000_IG226
       E9E9F5FFFF           jmp      G_M000_IG224
 
G_M000_IG327:                ;; offset=0x2471
       C5F810040A           vmovups  xmm0, xmmword ptr [rdx+rcx]
       62F37D083E0C0E04     vpcmpfalseub k1, xmm0, xmmword ptr [rsi+rcx]
       C5F898C9             kortestw k1, k1
       0F85F6F5FFFF         jne      G_M000_IG226
       E9A4F5FFFF           jmp      G_M000_IG221
 
G_M000_IG328:                ;; offset=0x248D
       33FF                 xor      edi, edi
       E95FDFFFFF           jmp      G_M000_IG41
 
G_M000_IG329:                ;; offset=0x2494
       48BF28CD8656B67F0000 mov      rdi, 0x7FB65686CD28
       E8BDA4F77C           call     CORINFO_HELP_NEWSFAST
       488BD8               mov      rbx, rax
       BFD1E70000           mov      edi, 0xE7D1
       48BE087F9256B67F0000 mov      rsi, 0x7FB656927F08
       FF156D3F61FF         call     [CORINFO_HELP_STRCNS]
       4C8BE8               mov      r13, rax
       BF97EA0000           mov      edi, 0xEA97
       48BE087F9256B67F0000 mov      rsi, 0x7FB656927F08
       FF15553F61FF         call     [CORINFO_HELP_STRCNS]
       488BF0               mov      rsi, rax
       498BFD               mov      rdi, r13
       FF15A9D79AFE         call     [System.String:Concat(System.String,System.String):System.String]
       488BF0               mov      rsi, rax
       488BFB               mov      rdi, rbx
       FF159D3F61FF         call     [System.ArgumentException:.ctor(System.String):this]
       488BFB               mov      rdi, rbx
       E80DD1E27C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG330:                ;; offset=0x24F4
       488BFB               mov      rdi, rbx
       488B0B               mov      rcx, qword ptr [rbx]
       488B4978             mov      rcx, qword ptr [rcx+0x78]
       FF5110               call     [rcx+0x10]Lokad.Onnx.Tensor`1[float]:ToDenseTensor():Lokad.Onnx.DenseTensor`1[float]:this
       488BF8               mov      rdi, rax
       498BF5               mov      rsi, r13
       FF15C3CBCBFF         call     [Lokad.Onnx.Tensor`1[float]:CountedCopy[float](Lokad.Onnx.DenseTensor`1[float],Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE0               mov      r12, rax
       E9F6DEFFFF           jmp      G_M000_IG42
 
G_M000_IG331:                ;; offset=0x2515
       498BF6               mov      rsi, r14
       E8635CC6FF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BE0               mov      r12, rax
       E910DFFFFF           jmp      G_M000_IG45
 
G_M000_IG332:                ;; offset=0x2525
       33C9                 xor      rcx, rcx
       48898D60FCFFFF       mov      bword ptr [rbp-0x3A0], rcx
       4533C0               xor      r8d, r8d
       44898578FDFFFF       mov      dword ptr [rbp-0x288], r8d
       488B8D60FCFFFF       mov      rcx, bword ptr [rbp-0x3A0]
       448B8578FDFFFF       mov      r8d, dword ptr [rbp-0x288]
       E915DFFFFF           jmp      G_M000_IG46
 
G_M000_IG333:                ;; offset=0x254B
       33F6                 xor      rsi, rsi
       33D2                 xor      edx, edx
       48898D60FCFFFF       mov      bword ptr [rbp-0x3A0], rcx
       44898578FDFFFF       mov      dword ptr [rbp-0x288], r8d
       E921DFFFFF           jmp      G_M000_IG47
 
G_M000_IG334:                ;; offset=0x2562
       33FF                 xor      rdi, rdi
       33C0                 xor      eax, eax
       E935DFFFFF           jmp      G_M000_IG48
 
G_M000_IG335:                ;; offset=0x256B
       4883F804             cmp      rax, 4
       7332                 jae      SHORT G_M000_IG341
 
G_M000_IG336:                ;; offset=0x2571
       33D2                 xor      edx, edx
       488BF0               mov      rsi, rax
       4883E602             and      rsi, 2
       740A                 je       SHORT G_M000_IG338
 
G_M000_IG337:                ;; offset=0x257C
       0FB711               movzx    rdx, word  ptr [rcx]
       440FB707             movzx    r8, word  ptr [rdi]
       412BD0               sub      edx, r8d
 
G_M000_IG338:                ;; offset=0x2586
       A801                 test     al, 1
       740C                 je       SHORT G_M000_IG340
 
G_M000_IG339:                ;; offset=0x258A
       0FB60431             movzx    rax, byte  ptr [rcx+rsi]
       0FB60C37             movzx    rcx, byte  ptr [rdi+rsi]
       2BC1                 sub      eax, ecx
       0BD0                 or       edx, eax
 
G_M000_IG340:                ;; offset=0x2596
       85D2                 test     edx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E909F5FFFF           jmp      G_M000_IG230
 
G_M000_IG341:                ;; offset=0x25A3
       4883C0FC             add      rax, -4
       8B11                 mov      edx, dword ptr [rcx]
       2B17                 sub      edx, dword ptr [rdi]
       8B0C01               mov      ecx, dword ptr [rcx+rax]
       2B0C07               sub      ecx, dword ptr [rdi+rax]
       0BCA                 or       ecx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E9EEF4FFFF           jmp      G_M000_IG230
 
G_M000_IG342:                ;; offset=0x25BE
       62F17C4810040A       vmovups  zmm0, zmmword ptr [rdx+rcx]
       62F37D483E0C0E04     vpcmpfalseub k1, zmm0, zmmword ptr [rsi+rcx]
       C4E1F898C9           kortestq k1, k1
       0F8558F5FFFF         jne      G_M000_IG239
       E91EDFFFFF           jmp      G_M000_IG53
 
G_M000_IG343:                ;; offset=0x25DD
       C5FC10043A           vmovups  ymm0, ymmword ptr [rdx+rdi]
       62F37D283E0C3E04     vpcmpfalseub k1, ymm0, ymmword ptr [rsi+rdi]
       C4E1F998C9           kortestd k1, k1
       0F853BF5FFFF         jne      G_M000_IG239
       E912F5FFFF           jmp      G_M000_IG237
 
G_M000_IG344:                ;; offset=0x25FA
       C5F810040A           vmovups  xmm0, xmmword ptr [rdx+rcx]
       62F37D083E0C0E04     vpcmpfalseub k1, xmm0, xmmword ptr [rsi+rcx]
       C5F898C9             kortestw k1, k1
       0F851FF5FFFF         jne      G_M000_IG239
       E9CDF4FFFF           jmp      G_M000_IG234
 
G_M000_IG345:                ;; offset=0x2616
       33FF                 xor      edi, edi
       E90CDFFFFF           jmp      G_M000_IG56
 
G_M000_IG346:                ;; offset=0x261D
       48BF28CD8656B67F0000 mov      rdi, 0x7FB65686CD28
       E834A3F77C           call     CORINFO_HELP_NEWSFAST
       4C8BF0               mov      r14, rax
       BFB2E90000           mov      edi, 0xE9B2
       48BE087F9256B67F0000 mov      rsi, 0x7FB656927F08
       FF15E43D61FF         call     [CORINFO_HELP_STRCNS]
       4C8BE8               mov      r13, rax
       BF97EA0000           mov      edi, 0xEA97
       48BE087F9256B67F0000 mov      rsi, 0x7FB656927F08
       FF15CC3D61FF         call     [CORINFO_HELP_STRCNS]
       488BF0               mov      rsi, rax
       498BFD               mov      rdi, r13
       FF1520D69AFE         call     [System.String:Concat(System.String,System.String):System.String]
       488BF0               mov      rsi, rax
       498BFE               mov      rdi, r14
       FF15143E61FF         call     [System.ArgumentException:.ctor(System.String):this]
       498BFE               mov      rdi, r14
       E884CFE27C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG347:                ;; offset=0x267D
       498BFE               mov      rdi, r14
       498B0E               mov      rcx, qword ptr [r14]
       488B4978             mov      rcx, qword ptr [rcx+0x78]
       FF5110               call     [rcx+0x10]Lokad.Onnx.Tensor`1[float]:ToDenseTensor():Lokad.Onnx.DenseTensor`1[float]:this
       488BF8               mov      rdi, rax
       498BF5               mov      rsi, r13
       FF153ACACBFF         call     [Lokad.Onnx.Tensor`1[float]:CountedCopy[float](Lokad.Onnx.DenseTensor`1[float],Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE0               mov      r12, rax
       E9A3DEFFFF           jmp      G_M000_IG57
 
G_M000_IG348:                ;; offset=0x269E
       33FF                 xor      rdi, rdi
       33F6                 xor      esi, esi
       E9B1DEFFFF           jmp      G_M000_IG58
 
G_M000_IG349:                ;; offset=0x26A7
       FF154B5514FF         call     [System.ThrowHelper:ThrowArgumentOutOfRangeException()]
       CC                   int3     
 
G_M000_IG350:                ;; offset=0x26AE
       49BD80A8C045B67F0000 mov      r13, 0x7FB645C0A880
       E9FEDEFFFF           jmp      G_M000_IG59
 
G_M000_IG351:                ;; offset=0x26BD
       33C0                 xor      rax, rax
       33FF                 xor      edi, edi
       E913DFFFFF           jmp      G_M000_IG60
 
G_M000_IG352:                ;; offset=0x26C6
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E94CDFFFFF           jmp      G_M000_IG61
 
G_M000_IG353:                ;; offset=0x26CF
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E97CDFFFFF           jmp      G_M000_IG62
 
G_M000_IG354:                ;; offset=0x26D8
       33D2                 xor      rdx, rdx
       48899558FCFFFF       mov      bword ptr [rbp-0x3A8], rdx
       4533C0               xor      r8d, r8d
       44898570FDFFFF       mov      dword ptr [rbp-0x290], r8d
       488B9558FCFFFF       mov      rdx, bword ptr [rbp-0x3A8]
       448B8570FDFFFF       mov      r8d, dword ptr [rbp-0x290]
       E99EDFFFFF           jmp      G_M000_IG63
 
G_M000_IG355:                ;; offset=0x26FE
       48899558FCFFFF       mov      bword ptr [rbp-0x3A8], rdx
       44898570FDFFFF       mov      dword ptr [rbp-0x290], r8d
       498BF7               mov      rsi, r15
       48BF3834BA57B67F0000 mov      rdi, 0x7FB657BA3438
       E8625AC6FF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BC8               mov      r9, rax
       488B9558FCFFFF       mov      rdx, bword ptr [rbp-0x3A8]
       448B8570FDFFFF       mov      r8d, dword ptr [rbp-0x290]
       E981DFFFFF           jmp      G_M000_IG64
 
G_M000_IG356:                ;; offset=0x2734
       4983795000           cmp      gword ptr [r9+0x50], 0
       0F847FDFFFFF         je       G_M000_IG65
       4D8B5150             mov      r10, gword ptr [r9+0x50]
       4C8995C0FCFFFF       mov      gword ptr [rbp-0x340], r10
       E97ADFFFFF           jmp      G_M000_IG66
 
G_M000_IG357:                ;; offset=0x274F
       E9EFF3FFFF           jmp      G_M000_IG241
 
G_M000_IG358:                ;; offset=0x2754
       E9EAF3FFFF           jmp      G_M000_IG241
 
G_M000_IG359:                ;; offset=0x2759
       E9E5F3FFFF           jmp      G_M000_IG241
 
G_M000_IG360:                ;; offset=0x275E
       413B7208             cmp      esi, dword ptr [r10+0x08]
       0F83B2010000         jae      G_M000_IG380
       8BC6                 mov      eax, esi
       418B448210           mov      eax, dword ptr [r10+4*rax+0x10]
       E9FDF3FFFF           jmp      G_M000_IG244
 
G_M000_IG361:                ;; offset=0x2774
       33C9                 xor      rcx, rcx
       48898D48FCFFFF       mov      bword ptr [rbp-0x3B8], rcx
       33D2                 xor      edx, edx
       899560FDFFFF         mov      dword ptr [rbp-0x2A0], edx
       488B8D48FCFFFF       mov      rcx, bword ptr [rbp-0x3B8]
       8B9560FDFFFF         mov      edx, dword ptr [rbp-0x2A0]
       E9EDDFFFFF           jmp      G_M000_IG77
 
G_M000_IG362:                ;; offset=0x2797
       48898D48FCFFFF       mov      bword ptr [rbp-0x3B8], rcx
       899560FDFFFF         mov      dword ptr [rbp-0x2A0], edx
       488BF3               mov      rsi, rbx
       48BF3834BA57B67F0000 mov      rdi, 0x7FB657BA3438
       E8CA59C6FF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BC0               mov      r8, rax
       488B8D48FCFFFF       mov      rcx, bword ptr [rbp-0x3B8]
       8B9560FDFFFF         mov      edx, dword ptr [rbp-0x2A0]
       E9D7DFFFFF           jmp      G_M000_IG79
 
G_M000_IG363:                ;; offset=0x27CB
       4983785000           cmp      gword ptr [r8+0x50], 0
       0F84D5DFFFFF         je       G_M000_IG80
       4D8B4850             mov      r9, gword ptr [r8+0x50]
       4C898DB8FCFFFF       mov      gword ptr [rbp-0x348], r9
       E9D0DFFFFF           jmp      G_M000_IG81
 
G_M000_IG364:                ;; offset=0x27E6
       E9AEF3FFFF           jmp      G_M000_IG246
 
G_M000_IG365:                ;; offset=0x27EB
       E9A9F3FFFF           jmp      G_M000_IG246
 
G_M000_IG366:                ;; offset=0x27F0
       E9A4F3FFFF           jmp      G_M000_IG246
 
G_M000_IG367:                ;; offset=0x27F5
       413B7108             cmp      esi, dword ptr [r9+0x08]
       0F831B010000         jae      G_M000_IG380
       8BCE                 mov      ecx, esi
       418B4C8910           mov      ecx, dword ptr [r9+4*rcx+0x10]
       E9BCF3FFFF           jmp      G_M000_IG249
 
G_M000_IG368:                ;; offset=0x280B
       33C9                 xor      rcx, rcx
       33D2                 xor      edx, edx
       E955E0FFFF           jmp      G_M000_IG92
 
G_M000_IG369:                ;; offset=0x2814
       E9D3F3FFFF           jmp      G_M000_IG251
 
G_M000_IG370:                ;; offset=0x2819
       E9CEF3FFFF           jmp      G_M000_IG251
 
G_M000_IG371:                ;; offset=0x281E
       E9C9F3FFFF           jmp      G_M000_IG251
 
G_M000_IG372:                ;; offset=0x2823
       3B4708               cmp      eax, dword ptr [rdi+0x08]
       0F83EE000000         jae      G_M000_IG380
       8BF0                 mov      esi, eax
       8B4CB710             mov      ecx, dword ptr [rdi+4*rsi+0x10]
       E9E3F3FFFF           jmp      G_M000_IG254
 
G_M000_IG373:                ;; offset=0x2837
       837DD402             cmp      dword ptr [rbp-0x2C], 2
       0F8C08E1FFFF         jl       G_M000_IG106
       8B4870               mov      ecx, dword ptr [rax+0x70]
       448B45D4             mov      r8d, dword ptr [rbp-0x2C]
       413BC8               cmp      ecx, r8d
       410F4FC8             cmovg    ecx, r8d
       898D38FFFFFF         mov      dword ptr [rbp-0xC8], ecx
       E9F9E0FFFF           jmp      G_M000_IG107
 
G_M000_IG374:                ;; offset=0x285A
       BA56555555           mov      edx, 0x55555556
       8BC2                 mov      eax, edx
       41F7ED               imul     edx:eax, r13d
       8BC2                 mov      eax, edx
       C1E81F               shr      eax, 31
       03C2                 add      eax, edx
       8D0440               lea      eax, [rax+2*rax]
       442BE8               sub      r13d, eax
       0F851DE2FFFF         jne      G_M000_IG118
       E903E1FFFF           jmp      G_M000_IG108
 
G_M000_IG375:                ;; offset=0x287C
       33C0                 xor      eax, eax
       E927E1FFFF           jmp      G_M000_IG110
 
G_M000_IG376:                ;; offset=0x2883
       33C9                 xor      rcx, rcx
       33C0                 xor      eax, eax
       E93AE1FFFF           jmp      G_M000_IG111
 
G_M000_IG377:                ;; offset=0x288C
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E94BE1FFFF           jmp      G_M000_IG112
 
G_M000_IG378:                ;; offset=0x2895
       48BA882D8071AE7F0000 mov      rdx, 0x7FAE71802D88
       488B12               mov      rdx, gword ptr [rdx]
       4885D2               test     rdx, rdx
       7556                 jne      SHORT G_M000_IG379
       48BFB0449657B67F0000 mov      rdi, 0x7FB6579644B0
       E8AAA0F77C           call     CORINFO_HELP_NEWSFAST
       488BD0               mov      rdx, rax
       48899518FDFFFF       mov      gword ptr [rbp-0x2E8], rdx
       48BE702B8071AE7F0000 mov      rsi, 0x7FAE71802B70
       488B36               mov      rsi, gword ptr [rsi]
       488BFA               mov      rdi, rdx
       48BAD0A0B357B67F0000 mov      rdx, 0x7FB657B3A0D0
       FF1548659AFE         call     [System.MulticastDelegate:CtorClosed(System.Object,nint):this]
       48BF882D8071AE7F0000 mov      rdi, 0x7FAE71802D88
       488BB518FDFFFF       mov      rsi, gword ptr [rbp-0x2E8]
       E86AF799FD           call     CORINFO_HELP_ASSIGN_REF
       488B9518FDFFFF       mov      rdx, gword ptr [rbp-0x2E8]
 
G_M000_IG379:                ;; offset=0x28FD
       488BBD30FDFFFF       mov      rdi, gword ptr [rbp-0x2D0]
       488BF2               mov      rsi, rdx
       FF158B3DB9FF         call     [System.Linq.Enumerable:All[int](System.Collections.Generic.IEnumerable`1[int],System.Func`2[int,bool]):bool]
       85C0                 test     eax, eax
       0F8484E1FFFF         je       G_M000_IG119
       E912E1FFFF           jmp      G_M000_IG115
 
G_M000_IG380:                ;; offset=0x291A
       E8710E9AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG381:                ;; offset=0x2920
       4883EC38             sub      rsp, 56
 
G_M000_IG382:                ;; offset=0x2924
       33FF                 xor      rdi, rdi
       4889BDB0FDFFFF       mov      gword ptr [rbp-0x250], rdi
 
G_M000_IG383:                ;; offset=0x292D
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG384:                ;; offset=0x2935
       4883EC38             sub      rsp, 56
 
G_M000_IG385:                ;; offset=0x2939
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB598FCFFFF       mov      rsi, gword ptr [rbp-0x368]
       33D2                 xor      edx, edx
       FF15E3ADD5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       90                   nop      
 
G_M000_IG386:                ;; offset=0x2956
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG387:                ;; offset=0x295E
       4883EC38             sub      rsp, 56
 
G_M000_IG388:                ;; offset=0x2962
       33FF                 xor      rdi, rdi
       4889BDB0FDFFFF       mov      gword ptr [rbp-0x250], rdi
 
G_M000_IG389:                ;; offset=0x296B
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG390:                ;; offset=0x2973
       4883EC38             sub      rsp, 56
 
G_M000_IG391:                ;; offset=0x2977
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB5A0FCFFFF       mov      rsi, gword ptr [rbp-0x360]
       33D2                 xor      edx, edx
       FF15A5ADD5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       90                   nop      
 
G_M000_IG392:                ;; offset=0x2994
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG393:                ;; offset=0x299C
       4883EC38             sub      rsp, 56
 
G_M000_IG394:                ;; offset=0x29A0
       33FF                 xor      rdi, rdi
       4889BDB0FDFFFF       mov      gword ptr [rbp-0x250], rdi
 
G_M000_IG395:                ;; offset=0x29A9
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG396:                ;; offset=0x29B1
       4883EC38             sub      rsp, 56
 
G_M000_IG397:                ;; offset=0x29B5
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       4C8B37               mov      r14, gword ptr [rdi]
       4883BDA8FCFFFF00     cmp      gword ptr [rbp-0x358], 0
       750C                 jne      SHORT G_M000_IG399
 
G_M000_IG398:                ;; offset=0x29CC
       BF02000000           mov      edi, 2
       FF1571E5ADFF         call     [System.ThrowHelper:ThrowArgumentNullException(int)]
       CC                   int3     
 
G_M000_IG399:                ;; offset=0x29D8
       488BBDA8FCFFFF       mov      rdi, gword ptr [rbp-0x358]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
       FFCF                 dec      edi
       83CF0F               or       edi, 15
       33DB                 xor      ebx, ebx
       F30FBDDF             lzcnt    ebx, edi
       83F31F               xor      ebx, 31
       83C3FD               add      ebx, -3
       48BFE8ECE9D4B67F0000 mov      rdi, 0x7FB6D4E9ECE8
       48B8208881D5B67F0000 mov      rax, 0x7FB6D5818820
       FFD0                 call     rax
       833809               cmp      dword ptr [rax], 9
       7E0D                 jle      SHORT G_M000_IG400
       488B7808             mov      rdi, gword ptr [rax+0x08]
       488B4748             mov      rax, bword ptr [rdi+0x48]
       4885C0               test     rax, rax
       750A                 jne      SHORT G_M000_IG401
 
G_M000_IG400:                ;; offset=0x2A1B
       BF09000000           mov      edi, 9
       E83BEBEFFF           call     CORINFO_HELP_GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED
 
G_M000_IG401:                ;; offset=0x2A25
       488B4010             mov      rax, gword ptr [rax+0x10]
       4885C0               test     rax, rax
       7509                 jne      SHORT G_M000_IG403
 
G_M000_IG402:                ;; offset=0x2A2E
       498BFE               mov      rdi, r14
       FF15C1E6CBFF         call     [System.Buffers.SharedArrayPool`1[float]:InitializeTlsBucketsAndTrimming():System.Buffers.SharedArrayPoolThreadLocalArray[]:this]
 
G_M000_IG403:                ;; offset=0x2A37
       4533FF               xor      r15d, r15d
       41BD01000000         mov      r13d, 1
       395808               cmp      dword ptr [rax+0x08], ebx
       0F863C020000         jbe      G_M000_IG421
 
G_M000_IG404:                ;; offset=0x2A49
       41BF01000000         mov      r15d, 1
       488BBDA8FCFFFF       mov      rdi, gword ptr [rbp-0x358]
       BE10000000           mov      esi, 16
       C4E261F7F6           shlx     esi, esi, ebx
       397708               cmp      dword ptr [rdi+0x08], esi
       7448                 je       SHORT G_M000_IG406
 
G_M000_IG405:                ;; offset=0x2A65
       48BF28CD8656B67F0000 mov      rdi, 0x7FB65686CD28
       E8EC9EF77C           call     CORINFO_HELP_NEWSFAST
       488BD8               mov      rbx, rax
       FF1593E6CBFF         call     [System.SR:get_ArgumentException_BufferNotFromPool():System.String]
       4C8BE0               mov      r12, rax
       BF6D040000           mov      edi, 0x46D
       48BE00408355B67F0000 mov      rsi, 0x7FB655834000
       FF15933961FF         call     [CORINFO_HELP_STRCNS]
       488BD0               mov      rdx, rax
       498BF4               mov      rsi, r12
       488BFB               mov      rdi, rbx
       FF15A43A61FF         call     [System.ArgumentException:.ctor(System.String,System.String):this]
       488BFB               mov      rdi, rbx
       E854CBE27C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG406:                ;; offset=0x2AAD
       3B5808               cmp      ebx, dword ptr [rax+0x08]
       0F83A2020000         jae      G_M000_IG425
       8BFB                 mov      edi, ebx
       48C1E704             shl      rdi, 4
       4C8D643810           lea      r12, bword ptr [rax+rdi+0x10]
       498B0424             mov      rax, gword ptr [r12]
       48898590FCFFFF       mov      gword ptr [rbp-0x370], rax
       488BB5A8FCFFFF       mov      rsi, gword ptr [rbp-0x358]
       498BFC               mov      rdi, r12
       E885F599FD           call     CORINFO_HELP_ASSIGN_REF
       33FF                 xor      edi, edi
       41897C2408           mov      dword ptr [r12+0x08], edi
       4C8BA590FCFFFF       mov      r12, gword ptr [rbp-0x370]
       4D85E4               test     r12, r12
       0F8493010000         je       G_M000_IG421
 
G_M000_IG407:                ;; offset=0x2AF2
       498B7E10             mov      rdi, gword ptr [r14+0x10]
       3B5F08               cmp      ebx, dword ptr [rdi+0x08]
       0F8359020000         jae      G_M000_IG425
       8BF3                 mov      esi, ebx
       488B44F710           mov      rax, gword ptr [rdi+8*rsi+0x10]
       4885C0               test     rax, rax
       750B                 jne      SHORT G_M000_IG408
       498BFE               mov      rdi, r14
       8BF3                 mov      esi, ebx
       FF152AE6CBFF         call     [System.Buffers.SharedArrayPool`1[float]:CreatePerCorePartitions(int):System.Buffers.SharedArrayPoolPartitions:this]
 
G_M000_IG408:                ;; offset=0x2B16
       4C8B6808             mov      r13, gword ptr [rax+0x08]
       48BFF008DD57B67F0000 mov      rdi, 0x7FB657DD08F0
       FF151E4C9AFE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       803D4BA89BFD00       cmp      byte  ptr [(reloc 0x7fb65583b35c)], 0
       7412                 je       SHORT G_M000_IG409
       C5F877               vzeroupper 
       E8457E99FE           call     Interop+Sys:SchedGetCpu():int
       8BD0                 mov      edx, eax
       899598FDFFFF         mov      dword ptr [rbp-0x268], edx
       EB4B                 jmp      SHORT G_M000_IG411
 
G_M000_IG409:                ;; offset=0x2B45
       BF0A000000           mov      edi, 10
       FF1588E7E0FF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B4010               mov      eax, dword ptr [rax+0x10]
       898594FDFFFF         mov      dword ptr [rbp-0x26C], eax
       BF0A000000           mov      edi, 10
       FF1574E7E0FF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B9594FDFFFF         mov      edx, dword ptr [rbp-0x26C]
       8D4AFF               lea      ecx, [rdx-0x01]
       894810               mov      dword ptr [rax+0x10], ecx
       0FB7C2               movzx    rax, dx
       85C0                 test     eax, eax
       7510                 jne      SHORT G_M000_IG410
       FF1573E7E0FF         call     [System.Threading.ProcessorIdCache:RefreshCurrentProcessorId():int]
       8BD0                 mov      edx, eax
       899598FDFFFF         mov      dword ptr [rbp-0x268], edx
       EB09                 jmp      SHORT G_M000_IG411
 
G_M000_IG410:                ;; offset=0x2B87
       C1FA10               sar      edx, 16
       899598FDFFFF         mov      dword ptr [rbp-0x268], edx
 
G_M000_IG411:                ;; offset=0x2B90
       48BF9007DD57B67F0000 mov      rdi, 0x7FB657DD0790
       FF15A84B9AFE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       8B8598FDFFFF         mov      eax, dword ptr [rbp-0x268]
       33D2                 xor      edx, edx
       F735C2A79BFD         div      edx:eax, dword ptr [(reloc 0x7fb65583b350)]
       8BC2                 mov      eax, edx
       33C9                 xor      ecx, ecx
       E9B1000000           jmp      G_M000_IG418
 
G_M000_IG412:                ;; offset=0x2BB7
       413B4508             cmp      eax, dword ptr [r13+0x08]
       0F8397010000         jae      G_M000_IG425
       8985A0FDFFFF         mov      dword ptr [rbp-0x260], eax
       8BF8                 mov      edi, eax
       498B54FD10           mov      rdx, gword ptr [r13+8*rdi+0x10]
       48899588FCFFFF       mov      gword ptr [rbp-0x378], rdx
       3812                 cmp      byte  ptr [rdx], dl
       33F6                 xor      esi, esi
       89B590FDFFFF         mov      dword ptr [rbp-0x270], esi
       488BFA               mov      rdi, rdx
       FF15D863F6FF         call     [System.Threading.Monitor:Enter(System.Object)]
       488B8588FCFFFF       mov      rax, gword ptr [rbp-0x378]
       488B7808             mov      rdi, gword ptr [rax+0x08]
       8B4810               mov      ecx, dword ptr [rax+0x10]
       898D8CFDFFFF         mov      dword ptr [rbp-0x274], ecx
       394F08               cmp      dword ptr [rdi+0x08], ecx
       7635                 jbe      SHORT G_M000_IG414
       85C9                 test     ecx, ecx
       7545                 jne      SHORT G_M000_IG415
       33F6                 xor      esi, esi
       897014               mov      dword ptr [rax+0x14], esi
 
G_M000_IG413:                ;; offset=0x2C0A
       4863F1               movsxd   rsi, ecx
       488D7CF710           lea      rdi, bword ptr [rdi+8*rsi+0x10]
       498BF4               mov      rsi, r12
       E846F499FD           call     CORINFO_HELP_ASSIGN_REF
       8BBD8CFDFFFF         mov      edi, dword ptr [rbp-0x274]
       FFC7                 inc      edi
       488B8588FCFFFF       mov      rax, gword ptr [rbp-0x378]
       897810               mov      dword ptr [rax+0x10], edi
       C78590FDFFFF01000000 mov      dword ptr [rbp-0x270], 1
 
G_M000_IG414:                ;; offset=0x2C36
       488BF8               mov      rdi, rax
       FF15E95B9AFE         call     [System.Threading.Monitor:Exit(System.Object)]
       83BD90FDFFFF00       cmp      dword ptr [rbp-0x270], 0
       7404                 je       SHORT G_M000_IG416
       EB30                 jmp      SHORT G_M000_IG419
 
G_M000_IG415:                ;; offset=0x2C4A
       EBBE                 jmp      SHORT G_M000_IG413
 
G_M000_IG416:                ;; offset=0x2C4C
       8B85A0FDFFFF         mov      eax, dword ptr [rbp-0x260]
       FFC0                 inc      eax
       8BF8                 mov      edi, eax
       41397D08             cmp      dword ptr [r13+0x08], edi
       7502                 jne      SHORT G_M000_IG417
       33FF                 xor      edi, edi
 
G_M000_IG417:                ;; offset=0x2C5E
       8B8D9CFDFFFF         mov      ecx, dword ptr [rbp-0x264]
       FFC1                 inc      ecx
       8BC7                 mov      eax, edi
 
G_M000_IG418:                ;; offset=0x2C68
       898D9CFDFFFF         mov      dword ptr [rbp-0x264], ecx
       41394D08             cmp      dword ptr [r13+0x08], ecx
       0F8F3FFFFFFF         jg       G_M000_IG412
       EB08                 jmp      SHORT G_M000_IG420
 
G_M000_IG419:                ;; offset=0x2C7A
       41BD01000000         mov      r13d, 1
       EB03                 jmp      SHORT G_M000_IG421
 
G_M000_IG420:                ;; offset=0x2C82
       4533ED               xor      r13d, r13d
 
G_M000_IG421:                ;; offset=0x2C85
       48BFF8018071AE7F0000 mov      rdi, 0x7FAE718001F8
       4C8B27               mov      r12, gword ptr [rdi]
       4180BC249D00000000   cmp      byte  ptr [r12+0x9D], 0
       0F84BD000000         je       G_M000_IG426
 
G_M000_IG422:                ;; offset=0x2CA1
       488BBDA8FCFFFF       mov      rdi, gword ptr [rbp-0x358]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       0F84AC000000         je       G_M000_IG426
       488BBDA8FCFFFF       mov      rdi, gword ptr [rbp-0x358]
       FF15316073FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8985ACFDFFFF         mov      dword ptr [rbp-0x254], eax
       488BBDA8FCFFFF       mov      rdi, gword ptr [rbp-0x358]
       8B4F08               mov      ecx, dword ptr [rdi+0x08]
       898DA8FDFFFF         mov      dword ptr [rbp-0x258], ecx
       498BFE               mov      rdi, r14
       FF15126073FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       448BC0               mov      r8d, eax
       498BFC               mov      rdi, r12
       8B95ACFDFFFF         mov      edx, dword ptr [rbp-0x254]
       8B8DA8FDFFFF         mov      ecx, dword ptr [rbp-0x258]
       BE03000000           mov      esi, 3
       FF15CD3CF6FF         call     [System.Diagnostics.Tracing.EventSource:WriteEvent(int,int,int,int):this]
       4585FD               test     r15d, r13d
       755E                 jne      SHORT G_M000_IG426
       488BBDA8FCFFFF       mov      rdi, gword ptr [rbp-0x358]
       FF15E35F73FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       448BE8               mov      r13d, eax
       488BBDA8FCFFFF       mov      rdi, gword ptr [rbp-0x358]
       8B4708               mov      eax, dword ptr [rdi+0x08]
       8985A4FDFFFF         mov      dword ptr [rbp-0x25C], eax
       498BFE               mov      rdi, r14
       FF15C75F73FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       4585FF               test     r15d, r15d
       750E                 jne      SHORT G_M000_IG423
       41B8FFFFFFFF         mov      r8d, -1
       41B901000000         mov      r9d, 1
       EB06                 jmp      SHORT G_M000_IG424
 
G_M000_IG423:                ;; offset=0x2D3E
       448BC3               mov      r8d, ebx
       4533C9               xor      r9d, r9d
 
G_M000_IG424:                ;; offset=0x2D44
       498BFC               mov      rdi, r12
       418BF5               mov      esi, r13d
       8B95A4FDFFFF         mov      edx, dword ptr [rbp-0x25C]
       FF1532E4CBFF         call     [System.Buffers.ArrayPoolEventSource:BufferDropped(int,int,int,int,int):this]
       EB06                 jmp      SHORT G_M000_IG426
 
G_M000_IG425:                ;; offset=0x2D58
       E8330A9AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG426:                ;; offset=0x2D5E
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG427:                ;; offset=0x2D66
       4883EC38             sub      rsp, 56
 
G_M000_IG428:                ;; offset=0x2D6A
       488D7D88             lea      rdi, [rbp-0x78]
       FF15E457B9FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG429:                ;; offset=0x2D75
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG430:                ;; offset=0x2D7D
       4883EC38             sub      rsp, 56
 
G_M000_IG431:                ;; offset=0x2D81
       488D7DA0             lea      rdi, [rbp-0x60]
       FF15CD57B9FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG432:                ;; offset=0x2D8C
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG433:                ;; offset=0x2D94
       4883EC38             sub      rsp, 56
 
G_M000_IG434:                ;; offset=0x2D98
       488D7DB8             lea      rdi, [rbp-0x48]
       FF15B657B9FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG435:                ;; offset=0x2DA3
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG436:                ;; offset=0x2DAB
       4883EC38             sub      rsp, 56
 
G_M000_IG437:                ;; offset=0x2DAF
       80BD48FEFFFF00       cmp      byte  ptr [rbp-0x1B8], 0
       740D                 je       SHORT G_M000_IG439
 
G_M000_IG438:                ;; offset=0x2DB8
       488BBDD0FCFFFF       mov      rdi, gword ptr [rbp-0x330]
       FF15635A9AFE         call     [System.Threading.Monitor:Exit(System.Object)]
 
G_M000_IG439:                ;; offset=0x2DC5
       90                   nop      
 
G_M000_IG440:                ;; offset=0x2DC6
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG441:                ;; offset=0x2DCE
       4883EC38             sub      rsp, 56
 
G_M000_IG442:                ;; offset=0x2DD2
       488DBD80FEFFFF       lea      rdi, [rbp-0x180]
       FF157957B9FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG443:                ;; offset=0x2DE0
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG444:                ;; offset=0x2DE8
       4883EC38             sub      rsp, 56
 
G_M000_IG445:                ;; offset=0x2DEC
       488DBD98FEFFFF       lea      rdi, [rbp-0x168]
       FF155F57B9FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG446:                ;; offset=0x2DFA
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
; Total bytes of code 11778
