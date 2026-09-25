; Assembly listing for method Lokad.Onnx.Tensor`1[float]:RunBatchedFloatMatMul(Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],Lokad.Onnx.TensorExecutionOptions) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 106932
; 70 inlinees with PGO data; 153 single block inlinees; 3 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       55                   push     rbp
       4157                 push     r15
       4156                 push     r14
       4155                 push     r13
       4154                 push     r12
       53                   push     rbx
       4881EC98020000       sub      rsp, 664
       488DAC24C0020000     lea      rbp, [rsp+0x2C0]
       C4413857C0           vxorps   xmm8, xmm8, xmm8
       62717D087F45E0       vmovdqa32 xmmword ptr [rbp-0x200], xmm8
       62717D087F45E1       vmovdqa32 xmmword ptr [rbp-0x1F0], xmm8
       48B850FEFFFFFFFFFFFF mov      rax, -432
       C5797F4405D0         vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       C5797F4405E0         vmovdqa  xmmword ptr [rbp+rax-0x20], xmm8
       C5797F4405F0         vmovdqa  xmmword ptr [rbp+rax-0x10], xmm8
       4883C030             add      rax, 48
       75E8                 jne      SHORT  -5 instr
       488945D0             mov      qword ptr [rbp-0x30], rax
       4C8BFF               mov      r15, rdi
       488BDE               mov      rbx, rsi
       4C8BF2               mov      r14, rdx
 
G_M000_IG02:                ;; offset=0x005B
       48BF50067F11677A0000 mov      rdi, 0x7A67117F0650
       E81604F67C           call     CORINFO_HELP_NEWSFAST
       48898528FEFFFF       mov      gword ptr [rbp-0x1D8], rax
       488D7848             lea      rdi, bword ptr [rax+0x48]
       40383F               cmp      byte  ptr [rdi], dil
       488D7510             lea      rsi, [rbp+0x10]
       BA30000000           mov      edx, 48
       C5F877               vzeroupper 
       FF153EB29AFE         call     [CORINFO_HELP_BULK_WRITEBARRIER]
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       4C8B6850             mov      r13, gword ptr [rax+0x50]
       498BFF               mov      rdi, r15
       4885FF               test     rdi, rdi
       7415                 je       SHORT G_M000_IG04
 
G_M000_IG03:                ;; offset=0x009D
       48BE20405E11677A0000 mov      rsi, 0x7A67115E4020
       483937               cmp      qword ptr [rdi], rsi
       0F85FE1A0000         jne      G_M000_IG262
       33FF                 xor      rdi, rdi
 
G_M000_IG04:                ;; offset=0x00B2
       4885FF               test     rdi, rdi
       0F850D1B0000         jne      G_M000_IG263
 
G_M000_IG05:                ;; offset=0x00BB
       4D8BE7               mov      r12, r15
       4D85E4               test     r12, r12
       7414                 je       SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x00C3
       48BF20405E11677A0000 mov      rdi, 0x7A67115E4020
       49393C24             cmp      qword ptr [r12], rdi
       0F85041B0000         jne      G_M000_IG264
 
G_M000_IG07:                ;; offset=0x00D7
       4D85E4               test     r12, r12
       0F846B1C0000         je       G_M000_IG280
       41807C243C00         cmp      byte  ptr [r12+0x3C], 0
       0F855F1C0000         jne      G_M000_IG280
       498B7C2418           mov      rdi, gword ptr [r12+0x18]
       4885FF               test     rdi, rdi
       0F84F11A0000         je       G_M000_IG265
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       448B4708             mov      r8d, dword ptr [rdi+0x08]
 
G_M000_IG08:                ;; offset=0x0102
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       4885FF               test     rdi, rdi
       0F84011B0000         je       G_M000_IG266
       488D7710             lea      rsi, bword ptr [rdi+0x10]
       8B5708               mov      edx, dword ptr [rdi+0x08]
       48898DA8FDFFFF       mov      bword ptr [rbp-0x258], rcx
       44898560FEFFFF       mov      dword ptr [rbp-0x1A0], r8d
 
G_M000_IG09:                ;; offset=0x0125
       488BFE               mov      rdi, rsi
       8BF2                 mov      esi, edx
       33D2                 xor      edx, edx
       FF151E9F61FF         call     [Lokad.Onnx.ArrayUtilities:GetStrides(System.ReadOnlySpan`1[int],bool):int[]]
       4885C0               test     rax, rax
       0F84ED1A0000         je       G_M000_IG267
       488D7810             lea      rdi, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG10:                ;; offset=0x0142
       398560FEFFFF         cmp      dword ptr [rbp-0x1A0], eax
       0F85961B0000         jne      G_M000_IG278
       488B8DA8FDFFFF       mov      rcx, bword ptr [rbp-0x258]
       488BD1               mov      rdx, rcx
       488BF7               mov      rsi, rdi
       8BC0                 mov      eax, eax
       48C1E002             shl      rax, 2
       4883F808             cmp      rax, 8
       0F82C61A0000         jb       G_M000_IG268
 
G_M000_IG11:                ;; offset=0x016B
       483BD6               cmp      rdx, rsi
       7456                 je       SHORT G_M000_IG17
 
G_M000_IG12:                ;; offset=0x0170
       4883F840             cmp      rax, 64
       0F8201170000         jb       G_M000_IG208
 
G_M000_IG13:                ;; offset=0x017A
       33C9                 xor      ecx, ecx
       4883C0C0             add      rax, -64
       7428                 je       SHORT G_M000_IG16
 
G_M000_IG14:                ;; offset=0x0182
       62F17C481002         vmovups  zmm0, zmmword ptr [rdx]
       62F37D483E0E04       vpcmpfalseub k1, zmm0, zmmword ptr [rsi]
       C4E1F898C9           kortestq k1, k1
       0F8593170000         jne      G_M000_IG220
 
G_M000_IG15:                ;; offset=0x019D
       4883C140             add      rcx, 64
       483BC1               cmp      rax, rcx
       0F87DA1A0000         ja       G_M000_IG275
 
G_M000_IG16:                ;; offset=0x01AA
       62F17C48100402       vmovups  zmm0, zmmword ptr [rdx+rax]
       62F17D48740C06       vpcmpeqb k1, zmm0, zmmword ptr [rsi+rax]
       C4E1F898C9           kortestq k1, k1
       0F836A170000         jae      G_M000_IG220
 
G_M000_IG17:                ;; offset=0x01C6
       BF01000000           mov      edi, 1
 
G_M000_IG18:                ;; offset=0x01CB
       85FF                 test     edi, edi
       0F84781B0000         je       G_M000_IG280
       418B7C245C           mov      edi, dword ptr [r12+0x5C]
       413B7C2430           cmp      edi, dword ptr [r12+0x30]
       0F85081B0000         jne      G_M000_IG279
 
G_M000_IG19:                ;; offset=0x01E3
       4D8BFC               mov      r15, r12
 
G_M000_IG20:                ;; offset=0x01E6
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       4C8B6850             mov      r13, gword ptr [rax+0x50]
       488BFB               mov      rdi, rbx
       4885FF               test     rdi, rdi
       7415                 je       SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x01F9
       48BE20405E11677A0000 mov      rsi, 0x7A67115E4020
       483937               cmp      qword ptr [rdi], rsi
       0F85601B0000         jne      G_M000_IG281
       33FF                 xor      rdi, rdi
 
G_M000_IG22:                ;; offset=0x020E
       4885FF               test     rdi, rdi
       0F856F1B0000         jne      G_M000_IG282
 
G_M000_IG23:                ;; offset=0x0217
       4C8BE3               mov      r12, rbx
       4D85E4               test     r12, r12
       7414                 je       SHORT G_M000_IG25
 
G_M000_IG24:                ;; offset=0x021F
       48BF20405E11677A0000 mov      rdi, 0x7A67115E4020
       49393C24             cmp      qword ptr [r12], rdi
       0F85661B0000         jne      G_M000_IG283
 
G_M000_IG25:                ;; offset=0x0233
       4D85E4               test     r12, r12
       0F84CD1C0000         je       G_M000_IG299
       41807C243C00         cmp      byte  ptr [r12+0x3C], 0
       0F85C11C0000         jne      G_M000_IG299
       498B7C2418           mov      rdi, gword ptr [r12+0x18]
       4885FF               test     rdi, rdi
       0F84531B0000         je       G_M000_IG284
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       448B4708             mov      r8d, dword ptr [rdi+0x08]
 
G_M000_IG26:                ;; offset=0x025E
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       4885FF               test     rdi, rdi
       0F84631B0000         je       G_M000_IG285
       488D7710             lea      rsi, bword ptr [rdi+0x10]
       8B5708               mov      edx, dword ptr [rdi+0x08]
       48898DA0FDFFFF       mov      bword ptr [rbp-0x260], rcx
       4489855CFEFFFF       mov      dword ptr [rbp-0x1A4], r8d
 
G_M000_IG27:                ;; offset=0x0281
       488BFE               mov      rdi, rsi
       8BF2                 mov      esi, edx
       33D2                 xor      edx, edx
       FF15C29D61FF         call     [Lokad.Onnx.ArrayUtilities:GetStrides(System.ReadOnlySpan`1[int],bool):int[]]
       4885C0               test     rax, rax
       0F844F1B0000         je       G_M000_IG286
       488D7810             lea      rdi, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG28:                ;; offset=0x029E
       39855CFEFFFF         cmp      dword ptr [rbp-0x1A4], eax
       0F85F81B0000         jne      G_M000_IG297
       488B8DA0FDFFFF       mov      rcx, bword ptr [rbp-0x260]
       488BD1               mov      rdx, rcx
       488BF7               mov      rsi, rdi
       8BC0                 mov      eax, eax
       48C1E002             shl      rax, 2
       4883F808             cmp      rax, 8
       0F82281B0000         jb       G_M000_IG287
 
G_M000_IG29:                ;; offset=0x02C7
       483BD6               cmp      rdx, rsi
       7456                 je       SHORT G_M000_IG35
 
G_M000_IG30:                ;; offset=0x02CC
       4883F840             cmp      rax, 64
       0F8261160000         jb       G_M000_IG221
 
G_M000_IG31:                ;; offset=0x02D6
       33C9                 xor      ecx, ecx
       4883C0C0             add      rax, -64
       7428                 je       SHORT G_M000_IG34
 
G_M000_IG32:                ;; offset=0x02DE
       62F17C481002         vmovups  zmm0, zmmword ptr [rdx]
       62F37D483E0E04       vpcmpfalseub k1, zmm0, zmmword ptr [rsi]
       C4E1F898C9           kortestq k1, k1
       0F85F3160000         jne      G_M000_IG233
 
G_M000_IG33:                ;; offset=0x02F9
       4883C140             add      rcx, 64
       483BC1               cmp      rax, rcx
       0F873C1B0000         ja       G_M000_IG294
 
G_M000_IG34:                ;; offset=0x0306
       62F17C48100402       vmovups  zmm0, zmmword ptr [rdx+rax]
       62F17D48740C06       vpcmpeqb k1, zmm0, zmmword ptr [rsi+rax]
       C4E1F898C9           kortestq k1, k1
       0F83CA160000         jae      G_M000_IG233
 
G_M000_IG35:                ;; offset=0x0322
       BF01000000           mov      edi, 1
 
G_M000_IG36:                ;; offset=0x0327
       85FF                 test     edi, edi
       0F84DA1B0000         je       G_M000_IG299
       418B7C245C           mov      edi, dword ptr [r12+0x5C]
       413B7C2430           cmp      edi, dword ptr [r12+0x30]
       0F856A1B0000         jne      G_M000_IG298
 
G_M000_IG37:                ;; offset=0x033F
       498BDC               mov      rbx, r12
 
G_M000_IG38:                ;; offset=0x0342
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       4C8B6850             mov      r13, gword ptr [rax+0x50]
       4D8BE6               mov      r12, r14
       4D85E4               test     r12, r12
       7414                 je       SHORT G_M000_IG40
 
G_M000_IG39:                ;; offset=0x0355
       48BF20405E11677A0000 mov      rdi, 0x7A67115E4020
       49393C24             cmp      qword ptr [r12], rdi
       0F85C11B0000         jne      G_M000_IG300
 
G_M000_IG40:                ;; offset=0x0369
       4D85E4               test     r12, r12
       0F84281D0000         je       G_M000_IG316
       41807C243C00         cmp      byte  ptr [r12+0x3C], 0
       0F851C1D0000         jne      G_M000_IG316
       498B7C2418           mov      rdi, gword ptr [r12+0x18]
       4885FF               test     rdi, rdi
       0F84AE1B0000         je       G_M000_IG301
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       448B4708             mov      r8d, dword ptr [rdi+0x08]
 
G_M000_IG41:                ;; offset=0x0394
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       4885FF               test     rdi, rdi
       0F84BE1B0000         je       G_M000_IG302
       488D7710             lea      rsi, bword ptr [rdi+0x10]
       8B5708               mov      edx, dword ptr [rdi+0x08]
       48898D98FDFFFF       mov      bword ptr [rbp-0x268], rcx
       44898558FEFFFF       mov      dword ptr [rbp-0x1A8], r8d
 
G_M000_IG42:                ;; offset=0x03B7
       488BFE               mov      rdi, rsi
       8BF2                 mov      esi, edx
       33D2                 xor      edx, edx
       FF158C9C61FF         call     [Lokad.Onnx.ArrayUtilities:GetStrides(System.ReadOnlySpan`1[int],bool):int[]]
       4885C0               test     rax, rax
       0F84AA1B0000         je       G_M000_IG303
       488D7810             lea      rdi, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG43:                ;; offset=0x03D4
       398558FEFFFF         cmp      dword ptr [rbp-0x1A8], eax
       0F85531C0000         jne      G_M000_IG314
       488B8D98FDFFFF       mov      rcx, bword ptr [rbp-0x268]
       488BD1               mov      rdx, rcx
       488BF7               mov      rsi, rdi
       8BC0                 mov      eax, eax
       48C1E002             shl      rax, 2
       4883F808             cmp      rax, 8
       0F82831B0000         jb       G_M000_IG304
 
G_M000_IG44:                ;; offset=0x03FD
       483BD6               cmp      rdx, rsi
       7456                 je       SHORT G_M000_IG50
 
G_M000_IG45:                ;; offset=0x0402
       4883F840             cmp      rax, 64
       0F82E7150000         jb       G_M000_IG234
 
G_M000_IG46:                ;; offset=0x040C
       33C9                 xor      ecx, ecx
       4883C0C0             add      rax, -64
       7428                 je       SHORT G_M000_IG49
 
G_M000_IG47:                ;; offset=0x0414
       62F17C481002         vmovups  zmm0, zmmword ptr [rdx]
       62F37D483E0E04       vpcmpfalseub k1, zmm0, zmmword ptr [rsi]
       C4E1F898C9           kortestq k1, k1
       0F8579160000         jne      G_M000_IG246
 
G_M000_IG48:                ;; offset=0x042F
       4883C140             add      rcx, 64
       483BC1               cmp      rax, rcx
       0F87971B0000         ja       G_M000_IG311
 
G_M000_IG49:                ;; offset=0x043C
       62F17C48100402       vmovups  zmm0, zmmword ptr [rdx+rax]
       62F17D48740C06       vpcmpeqb k1, zmm0, zmmword ptr [rsi+rax]
       C4E1F898C9           kortestq k1, k1
       0F8350160000         jae      G_M000_IG246
 
G_M000_IG50:                ;; offset=0x0458
       BF01000000           mov      edi, 1
 
G_M000_IG51:                ;; offset=0x045D
       85FF                 test     edi, edi
       0F84351C0000         je       G_M000_IG316
       418B7C245C           mov      edi, dword ptr [r12+0x5C]
       413B7C2430           cmp      edi, dword ptr [r12+0x30]
       0F85C51B0000         jne      G_M000_IG315
 
G_M000_IG52:                ;; offset=0x0475
       4D8BF4               mov      r14, r12
       498B7710             mov      rsi, gword ptr [r15+0x10]
       4885F6               test     rsi, rsi
       0F84361C0000         je       G_M000_IG317
       488D7E10             lea      rdi, bword ptr [rsi+0x10]
       8B7608               mov      esi, dword ptr [rsi+0x08]
 
G_M000_IG53:                ;; offset=0x048C
       4889BD78FFFFFF       mov      bword ptr [rbp-0x88], rdi
       897580               mov      dword ptr [rbp-0x80], esi
       448B6D80             mov      r13d, dword ptr [rbp-0x80]
       4183C5FE             add      r13d, -2
       443B6D80             cmp      r13d, dword ptr [rbp-0x80]
       0F871C1C0000         ja       G_M000_IG318
       4C8BA578FFFFFF       mov      r12, bword ptr [rbp-0x88]
       4585ED               test     r13d, r13d
       0F84131C0000         je       G_M000_IG319
       4963F5               movsxd   rsi, r13d
       48BF58B14710677A0000 mov      rdi, 0x7A671047B158
       E8C600F67C           call     CORINFO_HELP_NEWARR_1_VC
       488BF8               mov      rdi, rax
       4889BDF0FDFFFF       mov      gword ptr [rbp-0x210], rdi
       418BD5               mov      edx, r13d
       48C1E202             shl      rdx, 2
       4883C710             add      rdi, 16
       498BF4               mov      rsi, r12
       FF1558AE9AFE         call     [System.SpanHelpers:Memmove(byref,byref,nuint)]
       4C8BADF0FDFFFF       mov      r13, gword ptr [rbp-0x210]
 
G_M000_IG54:                ;; offset=0x04EF
       4C89AD20FEFFFF       mov      gword ptr [rbp-0x1E0], r13
       498B7710             mov      rsi, gword ptr [r15+0x10]
       488BFE               mov      rdi, rsi
       4885FF               test     rdi, rdi
       0F84D41B0000         je       G_M000_IG320
       488D4710             lea      rax, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG55:                ;; offset=0x050D
       8D4FFE               lea      ecx, [rdi-0x02]
       3BCF                 cmp      ecx, edi
       0F83501E0000         jae      G_M000_IG352
       8BF9                 mov      edi, ecx
       448B2CB8             mov      r13d, dword ptr [rax+4*rdi]
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       44896838             mov      dword ptr [rax+0x38], r13d
       488BFE               mov      rdi, rsi
       4885FF               test     rdi, rdi
       0F84AE1B0000         je       G_M000_IG321
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG56:                ;; offset=0x053C
       8D57FF               lea      edx, [rdi-0x01]
       3BD7                 cmp      edx, edi
       0F83211E0000         jae      G_M000_IG352
       8BFA                 mov      edi, edx
       8B3CB9               mov      edi, dword ptr [rcx+4*rdi]
       89783C               mov      dword ptr [rax+0x3C], edi
       488B7B10             mov      rdi, gword ptr [rbx+0x10]
       4885FF               test     rdi, rdi
       0F84901B0000         je       G_M000_IG322
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG57:                ;; offset=0x0563
       8D57FF               lea      edx, [rdi-0x01]
       3BD7                 cmp      edx, edi
       0F83FA1D0000         jae      G_M000_IG352
       8BFA                 mov      edi, edx
       8B3CB9               mov      edi, dword ptr [rcx+4*rdi]
       897840               mov      dword ptr [rax+0x40], edi
       488BBD20FEFFFF       mov      rdi, gword ptr [rbp-0x1E0]
       448B6708             mov      r12d, dword ptr [rdi+0x08]
       4489A554FEFFFF       mov      dword ptr [rbp-0x1AC], r12d
       4885F6               test     rsi, rsi
       0F84641B0000         je       G_M000_IG323
       488D5610             lea      rdx, bword ptr [rsi+0x10]
       448B4608             mov      r8d, dword ptr [rsi+0x08]
 
G_M000_IG58:                ;; offset=0x0599
       4D8BCF               mov      r9, r15
       48BE20405E11677A0000 mov      rsi, 0x7A67115E4020
       493931               cmp      qword ptr [r9], rsi
       0F856C1B0000         jne      G_M000_IG324
       4533C9               xor      r9, r9
 
G_M000_IG59:                ;; offset=0x05B2
       4D85C9               test     r9, r9
       0F85961B0000         jne      G_M000_IG325
 
G_M000_IG60:                ;; offset=0x05BB
       4D8B5718             mov      r10, gword ptr [r15+0x18]
       4C8995E8FDFFFF       mov      gword ptr [rbp-0x218], r10
 
G_M000_IG61:                ;; offset=0x05C6
       8B8D54FEFFFF         mov      ecx, dword ptr [rbp-0x1AC]
       898D4CFEFFFF         mov      dword ptr [rbp-0x1B4], ecx
       48899588FDFFFF       mov      bword ptr [rbp-0x278], rdx
       44898548FEFFFF       mov      dword ptr [rbp-0x1B8], r8d
       8BF1                 mov      esi, ecx
       48BF58B14710677A0000 mov      rdi, 0x7A671047B158
       E89FFFF57C           call     CORINFO_HELP_NEWARR_1_VC
       488BC8               mov      rcx, rax
       33F6                 xor      esi, esi
       8BBD4CFEFFFF         mov      edi, dword ptr [rbp-0x1B4]
       85FF                 test     edi, edi
       7E5F                 jle      SHORT G_M000_IG71
 
G_M000_IG62:                ;; offset=0x0600
       4C8B95E8FDFFFF       mov      r10, gword ptr [rbp-0x218]
       4D85D2               test     r10, r10
       0F84AB140000         je       G_M000_IG248
 
G_M000_IG63:                ;; offset=0x0610
       448B8548FEFFFF       mov      r8d, dword ptr [rbp-0x1B8]
       413BF8               cmp      edi, r8d
       0F8F561B0000         jg       G_M000_IG328
 
G_M000_IG64:                ;; offset=0x0620
       41397A08             cmp      dword ptr [r10+0x08], edi
       0F8C471B0000         jl       G_M000_IG327
 
G_M000_IG65:                ;; offset=0x062A
       397908               cmp      dword ptr [rcx+0x08], edi
       0F8C391B0000         jl       G_M000_IG326
 
G_M000_IG66:                ;; offset=0x0633
       4183C0FE             add      r8d, -2
 
G_M000_IG67:                ;; offset=0x0637
       413BF0               cmp      esi, r8d
       7D13                 jge      SHORT G_M000_IG69
 
G_M000_IG68:                ;; offset=0x063C
       8BC6                 mov      eax, esi
       488B9588FDFFFF       mov      rdx, bword ptr [rbp-0x278]
       833C8201             cmp      dword ptr [rdx+4*rax], 1
       0F8560140000         jne      G_M000_IG247
 
G_M000_IG69:                ;; offset=0x064F
       33C0                 xor      eax, eax
 
G_M000_IG70:                ;; offset=0x0651
       448BCE               mov      r9d, esi
       4289448910           mov      dword ptr [rcx+4*r9+0x10], eax
       FFC6                 inc      esi
       3BF7                 cmp      esi, edi
       7CD8                 jl       SHORT G_M000_IG67
 
G_M000_IG71:                ;; offset=0x065F
       48898D18FEFFFF       mov      gword ptr [rbp-0x1E8], rcx
       4489A544FEFFFF       mov      dword ptr [rbp-0x1BC], r12d
       488B7310             mov      rsi, gword ptr [rbx+0x10]
       4885F6               test     rsi, rsi
       0F84171B0000         je       G_M000_IG330
       488D4E10             lea      rcx, bword ptr [rsi+0x10]
       8B5608               mov      edx, dword ptr [rsi+0x08]
 
G_M000_IG72:                ;; offset=0x0681
       4C8BC3               mov      r8, rbx
       48BE20405E11677A0000 mov      rsi, 0x7A67115E4020
       493930               cmp      qword ptr [r8], rsi
       0F851D1B0000         jne      G_M000_IG331
       4533C0               xor      r8, r8
 
G_M000_IG73:                ;; offset=0x069A
       4D85C0               test     r8, r8
       0F85451B0000         jne      G_M000_IG332
 
G_M000_IG74:                ;; offset=0x06A3
       4C8B4B18             mov      r9, gword ptr [rbx+0x18]
       4C898DE0FDFFFF       mov      gword ptr [rbp-0x220], r9
 
G_M000_IG75:                ;; offset=0x06AE
       8B8544FEFFFF         mov      eax, dword ptr [rbp-0x1BC]
       89853CFEFFFF         mov      dword ptr [rbp-0x1C4], eax
       48898D78FDFFFF       mov      bword ptr [rbp-0x288], rcx
       899538FEFFFF         mov      dword ptr [rbp-0x1C8], edx
       8BF0                 mov      esi, eax
       48BF58B14710677A0000 mov      rdi, 0x7A671047B158
       E8B8FEF57C           call     CORINFO_HELP_NEWARR_1_VC
       488BF8               mov      rdi, rax
       33F6                 xor      esi, esi
       8B853CFEFFFF         mov      eax, dword ptr [rbp-0x1C4]
       85C0                 test     eax, eax
       7E5C                 jle      SHORT G_M000_IG85
 
G_M000_IG76:                ;; offset=0x06E7
       4C8B8DE0FDFFFF       mov      r9, gword ptr [rbp-0x220]
       4D85C9               test     r9, r9
       0F841A140000         je       G_M000_IG253
 
G_M000_IG77:                ;; offset=0x06F7
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       3BC2                 cmp      eax, edx
       0F8F081B0000         jg       G_M000_IG335
 
G_M000_IG78:                ;; offset=0x0705
       41394108             cmp      dword ptr [r9+0x08], eax
       0F8CF91A0000         jl       G_M000_IG334
 
G_M000_IG79:                ;; offset=0x070F
       394708               cmp      dword ptr [rdi+0x08], eax
       0F8CEB1A0000         jl       G_M000_IG333
 
G_M000_IG80:                ;; offset=0x0718
       83C2FE               add      edx, -2
 
G_M000_IG81:                ;; offset=0x071B
       3BF2                 cmp      esi, edx
       7D14                 jge      SHORT G_M000_IG83
 
G_M000_IG82:                ;; offset=0x071F
       8BCE                 mov      ecx, esi
       4C8B8578FDFFFF       mov      r8, bword ptr [rbp-0x288]
       41833C8801           cmp      dword ptr [r8+4*rcx], 1
       0F85D2130000         jne      G_M000_IG252
 
G_M000_IG83:                ;; offset=0x0733
       33C9                 xor      ecx, ecx
 
G_M000_IG84:                ;; offset=0x0735
       448BD6               mov      r10d, esi
       42894C9710           mov      dword ptr [rdi+4*r10+0x10], ecx
       FFC6                 inc      esi
       3BF0                 cmp      esi, eax
       7CD8                 jl       SHORT G_M000_IG81
 
G_M000_IG85:                ;; offset=0x0743
       4889BD10FEFFFF       mov      gword ptr [rbp-0x1F0], rdi
       418BC4               mov      eax, r12d
       498B7610             mov      rsi, gword ptr [r14+0x10]
       4885F6               test     rsi, rsi
       0F84CE1A0000         je       G_M000_IG337
       488D4E10             lea      rcx, bword ptr [rsi+0x10]
       8B5608               mov      edx, dword ptr [rsi+0x08]
 
G_M000_IG86:                ;; offset=0x0761
       898534FEFFFF         mov      dword ptr [rbp-0x1CC], eax
       48898D70FDFFFF       mov      bword ptr [rbp-0x290], rcx
       899530FEFFFF         mov      dword ptr [rbp-0x1D0], edx
       4D8B4618             mov      r8, gword ptr [r14+0x18]
       4C8985D8FDFFFF       mov      gword ptr [rbp-0x228], r8
       8BF0                 mov      esi, eax
       48BF58B14710677A0000 mov      rdi, 0x7A671047B158
       E800FEF57C           call     CORINFO_HELP_NEWARR_1_VC
       4C8BC0               mov      r8, rax
       33C0                 xor      eax, eax
       8B9534FEFFFF         mov      edx, dword ptr [rbp-0x1CC]
       85D2                 test     edx, edx
       7E5C                 jle      SHORT G_M000_IG96
 
G_M000_IG87:                ;; offset=0x079F
       488BBDD8FDFFFF       mov      rdi, gword ptr [rbp-0x228]
       4885FF               test     rdi, rdi
       0F84B5130000         je       G_M000_IG258
 
G_M000_IG88:                ;; offset=0x07AF
       8BB530FEFFFF         mov      esi, dword ptr [rbp-0x1D0]
       3BD6                 cmp      edx, esi
       0F8F7E1A0000         jg       G_M000_IG340
 
G_M000_IG89:                ;; offset=0x07BD
       395708               cmp      dword ptr [rdi+0x08], edx
       0F8C701A0000         jl       G_M000_IG339
 
G_M000_IG90:                ;; offset=0x07C6
       41395008             cmp      dword ptr [r8+0x08], edx
       0F8C611A0000         jl       G_M000_IG338
 
G_M000_IG91:                ;; offset=0x07D0
       83C6FE               add      esi, -2
 
G_M000_IG92:                ;; offset=0x07D3
       3BC6                 cmp      eax, esi
       7D14                 jge      SHORT G_M000_IG94
 
G_M000_IG93:                ;; offset=0x07D7
       8BC8                 mov      ecx, eax
       4C8B8D70FDFFFF       mov      r9, bword ptr [rbp-0x290]
       41833C8901           cmp      dword ptr [r9+4*rcx], 1
       0F856E130000         jne      G_M000_IG257
 
G_M000_IG94:                ;; offset=0x07EB
       33C9                 xor      ecx, ecx
 
G_M000_IG95:                ;; offset=0x07ED
       448BD0               mov      r10d, eax
       43894C9010           mov      dword ptr [r8+4*r10+0x10], ecx
       FFC0                 inc      eax
       3BC2                 cmp      eax, edx
       7CD8                 jl       SHORT G_M000_IG92
 
G_M000_IG96:                ;; offset=0x07FB
       4C898508FEFFFF       mov      gword ptr [rbp-0x1F8], r8
       488BBD20FEFFFF       mov      rdi, gword ptr [rbp-0x1E0]
       488D4710             lea      rax, bword ptr [rdi+0x10]
       418BD4               mov      edx, r12d
       BE01000000           mov      esi, 1
       85D2                 test     edx, edx
       7E0E                 jle      SHORT G_M000_IG99
 
G_M000_IG97:                ;; offset=0x0819
       33C9                 xor      ecx, ecx
 
G_M000_IG98:                ;; offset=0x081B
       0FAF3408             imul     esi, dword ptr [rax+rcx]
       4883C104             add      rcx, 4
       FFCA                 dec      edx
       75F4                 jne      SHORT G_M000_IG98
 
G_M000_IG99:                ;; offset=0x0827
       89B51CFFFFFF         mov      dword ptr [rbp-0xE4], esi
       8975D4               mov      dword ptr [rbp-0x2C], esi
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       83787002             cmp      dword ptr [rax+0x70], 2
       0F8D131A0000         jge      G_M000_IG342
 
G_M000_IG100:                ;; offset=0x0841
       C78538FFFFFF01000000 mov      dword ptr [rbp-0xC8], 1
 
G_M000_IG101:                ;; offset=0x084B
       488D5048             lea      rdx, bword ptr [rax+0x48]
       4C8B4A10             mov      r9, gword ptr [rdx+0x10]
       440FB6522C           movzx    r10, byte  ptr [rdx+0x2C]
       0FB6522D             movzx    rdx, byte  ptr [rdx+0x2D]
       4585D2               test     r10d, r10d
       0F8427010000         je       G_M000_IG112
       85D2                 test     edx, edx
       0F841F010000         je       G_M000_IG112
       41F6C501             test     r13b, 1
       0F85001A0000         jne      G_M000_IG343
 
G_M000_IG102:                ;; offset=0x0877
       498BF9               mov      rdi, r9
       488BF3               mov      rsi, rbx
       FF15F51CCCFF         call     [Lokad.Onnx.GraphPacking:ResolvePacked(System.Collections.Generic.IReadOnlyDictionary`2[float[],Lokad.Onnx.PackedMatMulWeight],Lokad.Onnx.Tensor`1[float]):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE8               mov      r13, rax
       4D85ED               test     r13, r13
       0F84FD000000         je       G_M000_IG112
 
G_M000_IG103:                ;; offset=0x088F
       498B7D10             mov      rdi, gword ptr [r13+0x10]
       488BC7               mov      rax, rdi
       4885C0               test     rax, rax
       0F84FA190000         je       G_M000_IG344
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG104:                ;; offset=0x08A2
       83F802               cmp      eax, 2
       0F85E1000000         jne      G_M000_IG112
       488BC7               mov      rax, rdi
       4885C0               test     rax, rax
       0F84E9190000         je       G_M000_IG345
       488D4810             lea      rcx, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG105:                ;; offset=0x08BE
       85C0                 test     eax, eax
       0F84A21A0000         je       G_M000_IG352
       8B01                 mov      eax, dword ptr [rcx]
       4885FF               test     rdi, rdi
       0F84D8190000         je       G_M000_IG346
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG106:                ;; offset=0x08D8
       83FF01               cmp      edi, 1
       0F86871A0000         jbe      G_M000_IG352
       8B7904               mov      edi, dword ptr [rcx+0x04]
       85C0                 test     eax, eax
       0F8EA0000000         jle      G_M000_IG112
       85FF                 test     edi, edi
       0F8E98000000         jle      G_M000_IG112
       3D00100000           cmp      eax, 0x1000
       0F8D8D000000         jge      G_M000_IG112
       4898                 cdqe     
       4863FF               movsxd   rdi, edi
       480FAFF8             imul     rdi, rax
       4881FF00000008       cmp      rdi, 0x8000000
       7F7B                 jg       SHORT G_M000_IG112
 
G_M000_IG107:                ;; offset=0x0911
       4D85ED               test     r13, r13
       747B                 je       SHORT G_M000_IG113
 
G_M000_IG108:                ;; offset=0x091A
       837DD401             cmp      dword ptr [rbp-0x2C], 1
       0F858E190000         jne      G_M000_IG347
 
G_M000_IG109:                ;; offset=0x0924
       8B8D38FFFFFF         mov      ecx, dword ptr [rbp-0xC8]
       890C24               mov      dword ptr [rsp], ecx
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       8B5038               mov      edx, dword ptr [rax+0x38]
       89542408             mov      dword ptr [rsp+0x08], edx
       8B503C               mov      edx, dword ptr [rax+0x3C]
       89542410             mov      dword ptr [rsp+0x10], edx
       8B5040               mov      edx, dword ptr [rax+0x40]
       89542418             mov      dword ptr [rsp+0x18], edx
       4C896C2420           mov      gword ptr [rsp+0x20], r13
       498BFF               mov      rdi, r15
       498BF6               mov      rsi, r14
       488B9520FEFFFF       mov      rdx, gword ptr [rbp-0x1E0]
       488B8D18FEFFFF       mov      rcx, gword ptr [rbp-0x1E8]
       4C8B8508FEFFFF       mov      r8, gword ptr [rbp-0x1F8]
       448B8D1CFFFFFF       mov      r9d, dword ptr [rbp-0xE4]
       FF154A31CCFF         call     [Lokad.Onnx.Tensor`1[float]:RunPackedBatches(Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],int[],int[],int[],int,int,int,int,int,Lokad.Onnx.DenseTensor`1[float])]
 
G_M000_IG110:                ;; offset=0x0976
       90                   nop      
 
G_M000_IG111:                ;; offset=0x0977
       C5F877               vzeroupper 
       4881C498020000       add      rsp, 664
       5B                   pop      rbx
       415C                 pop      r12
       415D                 pop      r13
       415E                 pop      r14
       415F                 pop      r15
       5D                   pop      rbp
       C3                   ret      
 
G_M000_IG112:                ;; offset=0x098C
       4533ED               xor      r13, r13
       EB80                 jmp      SHORT G_M000_IG107
                            align    [0 bytes for IG127]
 
G_M000_IG113:                ;; offset=0x0991
       498BFF               mov      rdi, r15
       FF15DE2DCCFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       48898568FFFFFF       mov      gword ptr [rbp-0x98], rax
       48899570FFFFFF       mov      qword ptr [rbp-0x90], rdx
       488DBD68FFFFFF       lea      rdi, [rbp-0x98]
       488D75B8             lea      rsi, [rbp-0x48]
       FF1517B6B8FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG114:                ;; offset=0x09BA
       488BFB               mov      rdi, rbx
       FF15B52DCCFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       48898568FFFFFF       mov      gword ptr [rbp-0x98], rax
       48899570FFFFFF       mov      qword ptr [rbp-0x90], rdx
       488DBD68FFFFFF       lea      rdi, [rbp-0x98]
       488D75A0             lea      rsi, [rbp-0x60]
       FF15EEB5B8FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG115:                ;; offset=0x09E3
       498BFE               mov      rdi, r14
       393F                 cmp      dword ptr [rdi], edi
       FF158A2DCCFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       48898568FFFFFF       mov      gword ptr [rbp-0x98], rax
       48899570FFFFFF       mov      qword ptr [rbp-0x90], rdx
       488DBD68FFFFFF       lea      rdi, [rbp-0x98]
       488D7588             lea      rsi, [rbp-0x78]
       FF15C3B5B8FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG116:                ;; offset=0x0A0E
       488B75C0             mov      rsi, qword ptr [rbp-0x40]
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       48897020             mov      qword ptr [rax+0x20], rsi
       488B75A8             mov      rsi, qword ptr [rbp-0x58]
       48897028             mov      qword ptr [rax+0x28], rsi
       488B7590             mov      rsi, qword ptr [rbp-0x70]
       48897030             mov      qword ptr [rax+0x30], rsi
       8B9D38FFFFFF         mov      ebx, dword ptr [rbp-0xC8]
       83FB01               cmp      ebx, 1
       0F8F7B060000         jg       G_M000_IG162
 
G_M000_IG117:                ;; offset=0x0A3C
       488B7020             mov      rsi, qword ptr [rax+0x20]
       4889B560FFFFFF       mov      qword ptr [rbp-0xA0], rsi
       488B7028             mov      rsi, qword ptr [rax+0x28]
       4889B558FFFFFF       mov      qword ptr [rbp-0xA8], rsi
       488B7030             mov      rsi, qword ptr [rax+0x30]
       4889B550FFFFFF       mov      qword ptr [rbp-0xB0], rsi
       4489A54CFFFFFF       mov      dword ptr [rbp-0xB4], r12d
       8BB54CFFFFFF         mov      esi, dword ptr [rbp-0xB4]
       48BF58B14710677A0000 mov      rdi, 0x7A671047B158
       E817FBF57C           call     CORINFO_HELP_NEWARR_1_VC
       48898500FEFFFF       mov      gword ptr [rbp-0x200], rax
       33C9                 xor      ecx, ecx
       898D48FFFFFF         mov      dword ptr [rbp-0xB8], ecx
 
G_M000_IG118:                ;; offset=0x0A88
       898D44FFFFFF         mov      dword ptr [rbp-0xBC], ecx
 
G_M000_IG119:                ;; offset=0x0A8E
       898D40FFFFFF         mov      dword ptr [rbp-0xC0], ecx
 
G_M000_IG120:                ;; offset=0x0A94
       898D3CFFFFFF         mov      dword ptr [rbp-0xC4], ecx
       448B45D4             mov      r8d, dword ptr [rbp-0x2C]
       4439853CFFFFFF       cmp      dword ptr [rbp-0xC4], r8d
       0F8C50010000         jl       G_M000_IG129
       E9C10C0000           jmp      G_M000_IG190
 
G_M000_IG121:                ;; offset=0x0AB0
       E84A190000           call     G_M000_IG368
       90                   nop      
 
G_M000_IG122:                ;; offset=0x0AB6
       BA56555555           mov      edx, 0x55555556
       8BC2                 mov      eax, edx
       F7AD18FFFFFF         imul     edx:eax, dword ptr [rbp-0xE8]
       448BCA               mov      r9d, edx
       41C1E91F             shr      r9d, 31
       4403CA               add      r9d, edx
       478D0C49             lea      r9d, [r9+2*r9]
       448BB518FFFFFF       mov      r14d, dword ptr [rbp-0xE8]
       452BF1               sub      r14d, r9d
       0F84640B0000         je       G_M000_IG184
 
G_M000_IG123:                ;; offset=0x0AE1
       4533C9               xor      r9d, r9d
 
G_M000_IG124:                ;; offset=0x0AE4
       8B8D94FEFFFF         mov      ecx, dword ptr [rbp-0x16C]
       3B8D18FFFFFF         cmp      ecx, dword ptr [rbp-0xE8]
       0F85A80B0000         jne      G_M000_IG186
 
G_M000_IG125:                ;; offset=0x0AF6
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG126:                ;; offset=0x0AFF
       448BA54CFFFFFF       mov      r12d, dword ptr [rbp-0xB4]
       418D7C24FF           lea      edi, [r12-0x01]
       85FF                 test     edi, edi
       0F8CA8000000         jl       G_M000_IG128
 
G_M000_IG127:                ;; offset=0x0B13
       488B8D00FEFFFF       mov      rcx, gword ptr [rbp-0x200]
       8B7108               mov      esi, dword ptr [rcx+0x08]
       8BDE                 mov      ebx, esi
       3BFB                 cmp      edi, ebx
       0F83440C0000         jae      G_M000_IG189
       8BF7                 mov      esi, edi
       488D54B110           lea      rdx, bword ptr [rcx+4*rsi+0x10]
       8B02                 mov      eax, dword ptr [rdx]
       FFC0                 inc      eax
       8902                 mov      dword ptr [rdx], eax
       4C8B8518FEFFFF       mov      r8, gword ptr [rbp-0x1E8]
       413B7808             cmp      edi, dword ptr [r8+0x08]
       0F83260C0000         jae      G_M000_IG189
       418B54B010           mov      edx, dword ptr [r8+4*rsi+0x10]
       039548FFFFFF         add      edx, dword ptr [rbp-0xB8]
       899548FFFFFF         mov      dword ptr [rbp-0xB8], edx
       4C8B8D10FEFFFF       mov      r9, gword ptr [rbp-0x1F0]
       413B7908             cmp      edi, dword ptr [r9+0x08]
       0F83040C0000         jae      G_M000_IG189
       418B54B110           mov      edx, dword ptr [r9+4*rsi+0x10]
       039544FFFFFF         add      edx, dword ptr [rbp-0xBC]
       899544FFFFFF         mov      dword ptr [rbp-0xBC], edx
       4C8B9508FEFFFF       mov      r10, gword ptr [rbp-0x1F8]
       413B7A08             cmp      edi, dword ptr [r10+0x08]
       0F83E20B0000         jae      G_M000_IG189
       418B54B210           mov      edx, dword ptr [r10+4*rsi+0x10]
       039540FFFFFF         add      edx, dword ptr [rbp-0xC0]
       899540FFFFFF         mov      dword ptr [rbp-0xC0], edx
       3BFB                 cmp      edi, ebx
       0F83C90B0000         jae      G_M000_IG189
       3BFB                 cmp      edi, ebx
       0F83C10B0000         jae      G_M000_IG189
       488B9520FEFFFF       mov      rdx, gword ptr [rbp-0x1E0]
       3B44B210             cmp      eax, dword ptr [rdx+4*rsi+0x10]
       0F8D31040000         jge      G_M000_IG159
 
G_M000_IG128:                ;; offset=0x0BBB
       4C8B8518FEFFFF       mov      r8, gword ptr [rbp-0x1E8]
       4C8B9508FEFFFF       mov      r10, gword ptr [rbp-0x1F8]
       4C8B8D10FEFFFF       mov      r9, gword ptr [rbp-0x1F0]
       488B8D00FEFFFF       mov      rcx, gword ptr [rbp-0x200]
       488B9520FEFFFF       mov      rdx, gword ptr [rbp-0x1E0]
       8BBD3CFFFFFF         mov      edi, dword ptr [rbp-0xC4]
       FFC7                 inc      edi
       89BD3CFFFFFF         mov      dword ptr [rbp-0xC4], edi
       8B45D4               mov      eax, dword ptr [rbp-0x2C]
       39853CFFFFFF         cmp      dword ptr [rbp-0xC4], eax
       0F8D760B0000         jge      G_M000_IG190
 
G_M000_IG129:                ;; offset=0x0BFB
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       8B4838               mov      ecx, dword ptr [rax+0x38]
       898D18FFFFFF         mov      dword ptr [rbp-0xE8], ecx
       8B483C               mov      ecx, dword ptr [rax+0x3C]
       898D14FFFFFF         mov      dword ptr [rbp-0xEC], ecx
       8B4840               mov      ecx, dword ptr [rax+0x40]
       898D10FFFFFF         mov      dword ptr [rbp-0xF0], ecx
       48638D48FFFFFF       movsxd   rcx, dword ptr [rbp-0xB8]
       488BB560FFFFFF       mov      rsi, qword ptr [rbp-0xA0]
       488D0C8E             lea      rcx, [rsi+4*rcx]
       48898D08FFFFFF       mov      qword ptr [rbp-0xF8], rcx
       48638D44FFFFFF       movsxd   rcx, dword ptr [rbp-0xBC]
       4C8B9558FFFFFF       mov      r10, qword ptr [rbp-0xA8]
       498D0C8A             lea      rcx, [r10+4*rcx]
       48898D00FFFFFF       mov      qword ptr [rbp-0x100], rcx
       48638D40FFFFFF       movsxd   rcx, dword ptr [rbp-0xC0]
       4C8B9D50FFFFFF       mov      r11, qword ptr [rbp-0xB0]
       498D0C8B             lea      rcx, [r11+4*rcx]
       48898DF8FEFFFF       mov      qword ptr [rbp-0x108], rcx
       488D4848             lea      rcx, bword ptr [rax+0x48]
       4C8B21               mov      r12, gword ptr [rcx]
       0FB6592C             movzx    rbx, byte  ptr [rcx+0x2C]
       440FB6792D           movzx    r15, byte  ptr [rcx+0x2D]
 
G_M000_IG130:                ;; offset=0x0C78
       C5FE6F01             vmovdqu  ymm0, ymmword ptr [rcx]
       C5FE7F85C8FEFFFF     vmovdqu  ymmword ptr [rbp-0x138], ymm0
       C5FA6F4120           vmovdqu  xmm0, xmmword ptr [rcx+0x20]
       C5FA7F85E8FEFFFF     vmovdqu  xmmword ptr [rbp-0x118], xmm0
 
G_M000_IG131:                ;; offset=0x0C91
       83BD18FFFFFF30       cmp      dword ptr [rbp-0xE8], 48
       0F8D47050000         jge      G_M000_IG164
 
G_M000_IG132:                ;; offset=0x0C9E
       C5FE6F85C8FEFFFF     vmovdqu  ymm0, ymmword ptr [rbp-0x138]
       C5FE7F8598FEFFFF     vmovdqu  ymmword ptr [rbp-0x168], ymm0
       C5FA6F85E8FEFFFF     vmovdqu  xmm0, xmmword ptr [rbp-0x118]
       C5FA7F85B8FEFFFF     vmovdqu  xmmword ptr [rbp-0x148], xmm0
 
G_M000_IG133:                ;; offset=0x0CBE
       4533C9               xor      r9, r9
       4C898D88FEFFFF       mov      gword ptr [rbp-0x178], r9
       85DB                 test     ebx, ebx
       0F84EA020000         je       G_M000_IG158
 
G_M000_IG134:                ;; offset=0x0CD0
       4585FF               test     r15d, r15d
       740D                 je       SHORT G_M000_IG135
       83BD18FFFFFF01       cmp      dword ptr [rbp-0xE8], 1
       0F84F5050000         je       G_M000_IG165
 
G_M000_IG135:                ;; offset=0x0CE2
       4585FF               test     r15d, r15d
       0F84170A0000         je       G_M000_IG187
       83BD18FFFFFF02       cmp      dword ptr [rbp-0xE8], 2
       0F8C0A0A0000         jl       G_M000_IG187
       8B9518FFFFFF         mov      edx, dword ptr [rbp-0xE8]
       8B8D18FFFFFF         mov      ecx, dword ptr [rbp-0xE8]
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       C1EF1F               shr      edi, 31
       03BD18FFFFFF         add      edi, dword ptr [rbp-0xE8]
       83E7FE               and      edi, -2
       2BCF                 sub      ecx, edi
       2BD1                 sub      edx, ecx
       899594FEFFFF         mov      dword ptr [rbp-0x16C], edx
       BA56555555           mov      edx, 0x55555556
       8BC2                 mov      eax, edx
       F7AD18FFFFFF         imul     edx:eax, dword ptr [rbp-0xE8]
       8BCA                 mov      ecx, edx
       C1E91F               shr      ecx, 31
       03CA                 add      ecx, edx
       8D0C49               lea      ecx, [rcx+2*rcx]
       448BB518FFFFFF       mov      r14d, dword ptr [rbp-0xE8]
       442BF1               sub      r14d, ecx
       0F84D1050000         je       G_M000_IG166
 
G_M000_IG136:                ;; offset=0x0D47
       83BD94FEFFFF40       cmp      dword ptr [rbp-0x16C], 64
       0F8D32070000         jge      G_M000_IG174
 
G_M000_IG137:                ;; offset=0x0D54
       4585F6               test     r14d, r14d
       740D                 je       SHORT G_M000_IG139
 
G_M000_IG138:                ;; offset=0x0D59
       F68518FFFFFF01       test     byte  ptr [rbp-0xE8], 1
       0F850F030000         jne      G_M000_IG160
 
G_M000_IG139:                ;; offset=0x0D66
       4863BD14FFFFFF       movsxd   rdi, dword ptr [rbp-0xEC]
       48638510FFFFFF       movsxd   rax, dword ptr [rbp-0xF0]
       480FAFF8             imul     rdi, rax
       4881FF00000100       cmp      rdi, 0x10000
       0F8FF0020000         jg       G_M000_IG160
 
G_M000_IG140:                ;; offset=0x0D85
       448BBD14FFFFFF       mov      r15d, dword ptr [rbp-0xEC]
       440FAFBD10FFFFFF     imul     r15d, dword ptr [rbp-0xF0]
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       488B3F               mov      rdi, gword ptr [rdi]
       418BF7               mov      esi, r15d
       FF154EDBD5FF         call     [System.Buffers.SharedArrayPool`1[float]:Rent(int):float[]:this]
       488BD8               mov      rbx, rax
       4D85E4               test     r12, r12
       7424                 je       SHORT G_M000_IG141
       48BEA0697111677A0000 mov      rsi, 0x7A67117169A0
       49393424             cmp      qword ptr [r12], rsi
       0F8521080000         jne      G_M000_IG182
       4983C408             add      r12, 8
       4963F7               movsxd   rsi, r15d
       48C1E602             shl      rsi, 2
       F0                   lock     
       49013424             add      qword ptr [r12], rsi
 
G_M000_IG141:                ;; offset=0x0DD6
       48899DD0FDFFFF       mov      gword ptr [rbp-0x230], rbx
 
G_M000_IG142:                ;; offset=0x0DDD
       48899D88FEFFFF       mov      gword ptr [rbp-0x178], rbx
       4885DB               test     rbx, rbx
       0F8489010000         je       G_M000_IG154
 
G_M000_IG143:                ;; offset=0x0DED
       837B0800             cmp      dword ptr [rbx+0x08], 0
       0F847F010000         je       G_M000_IG154
       837B0800             cmp      dword ptr [rbx+0x08], 0
       0F86A5010000         jbe      G_M000_IG156
       4883C310             add      rbx, 16
       4C8BC3               mov      r8, rbx
 
G_M000_IG144:                ;; offset=0x0E08
       8BBD10FFFFFF         mov      edi, dword ptr [rbp-0xF0]
       8BB510FFFFFF         mov      esi, dword ptr [rbp-0xF0]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       C1FA1F               sar      edx, 31
       83E21F               and      edx, 31
       039510FFFFFF         add      edx, dword ptr [rbp-0xF0]
       83E2E0               and      edx, -32
       2BF2                 sub      esi, edx
       2BFE                 sub      edi, esi
       33F6                 xor      esi, esi
       85FF                 test     edi, edi
       0F8E83000000         jle      G_M000_IG148
 
G_M000_IG145:                ;; offset=0x0E37
       8BD6                 mov      edx, esi
       C1FA1F               sar      edx, 31
       83E21F               and      edx, 31
       03D6                 add      edx, esi
       C1FA05               sar      edx, 5
       0FAF9514FFFFFF       imul     edx, dword ptr [rbp-0xEC]
       C1E205               shl      edx, 5
       4863D2               movsxd   rdx, edx
       498D1490             lea      rdx, [r8+4*rdx]
       33C9                 xor      ecx, ecx
       83BD14FFFFFF00       cmp      dword ptr [rbp-0xEC], 0
       7E4F                 jle      SHORT G_M000_IG147
                            align    [0 bytes for IG146]
 
G_M000_IG146:                ;; offset=0x0E60
       448BC9               mov      r9d, ecx
       41C1E105             shl      r9d, 5
       4D63C9               movsxd   r9, r9d
       4E8D0C8A             lea      r9, [rdx+4*r9]
       8BC1                 mov      eax, ecx
       0FAF8510FFFFFF       imul     eax, dword ptr [rbp-0xF0]
       4898                 cdqe     
       48C1E002             shl      rax, 2
       48038500FFFFFF       add      rax, qword ptr [rbp-0x100]
       4C63D6               movsxd   r10, esi
       4A8D0490             lea      rax, [rax+4*r10]
       62F17E486F00         vmovdqu32 zmm0, zmmword ptr [rax]
       62F17E486F4801       vmovdqu32 zmm1, zmmword ptr [rax+0x40]
       62D17E487F01         vmovdqu32 zmmword ptr [r9], zmm0
       62D17E487F4901       vmovdqu32 zmmword ptr [r9+0x40], zmm1
       FFC1                 inc      ecx
       3B8D14FFFFFF         cmp      ecx, dword ptr [rbp-0xEC]
       7CB1                 jl       SHORT G_M000_IG146
 
G_M000_IG147:                ;; offset=0x0EAF
       83C620               add      esi, 32
       3BF7                 cmp      esi, edi
       7C81                 jl       SHORT G_M000_IG145
 
G_M000_IG148:                ;; offset=0x0EBA
       8BB510FFFFFF         mov      esi, dword ptr [rbp-0xF0]
       2BF7                 sub      esi, edi
       8BD7                 mov      edx, edi
       C1FA1F               sar      edx, 31
       83E21F               and      edx, 31
       03D7                 add      edx, edi
       C1FA05               sar      edx, 5
       0FAF9514FFFFFF       imul     edx, dword ptr [rbp-0xEC]
       C1E205               shl      edx, 5
       4863D2               movsxd   rdx, edx
       498D1490             lea      rdx, [r8+4*rdx]
       33C9                 xor      ecx, ecx
       83BD14FFFFFF00       cmp      dword ptr [rbp-0xEC], 0
       7E5E                 jle      SHORT G_M000_IG153
 
G_M000_IG149:                ;; offset=0x0EEB
       4863FF               movsxd   rdi, edi
       48C1E702             shl      rdi, 2
       EB0E                 jmp      SHORT G_M000_IG151
       0F1F4000             align    [4 bytes for IG152]
 
G_M000_IG150:                ;; offset=0x0EF8
       FFC1                 inc      ecx
       3B8D14FFFFFF         cmp      ecx, dword ptr [rbp-0xEC]
       7D47                 jge      SHORT G_M000_IG153
 
G_M000_IG151:                ;; offset=0x0F02
       448BC9               mov      r9d, ecx
       440FAF8D10FFFFFF     imul     r9d, dword ptr [rbp-0xF0]
       4D63C9               movsxd   r9, r9d
       49C1E102             shl      r9, 2
       4C038D00FFFFFF       add      r9, qword ptr [rbp-0x100]
       4C03CF               add      r9, rdi
       8BC1                 mov      eax, ecx
       0FAFC6               imul     eax, esi
       4898                 cdqe     
       488D0482             lea      rax, [rdx+4*rax]
       4533D2               xor      r10d, r10d
       85F6                 test     esi, esi
       7EC8                 jle      SHORT G_M000_IG150
 
G_M000_IG152:                ;; offset=0x0F30
       4D63DA               movsxd   r11, r10d
       C4817A100499         vmovss   xmm0, dword ptr [r9+4*r11]
       C4A17A110498         vmovss   dword ptr [rax+4*r11], xmm0
       41FFC2               inc      r10d
       443BD6               cmp      r10d, esi
       7CE9                 jl       SHORT G_M000_IG152
       EBAF                 jmp      SHORT G_M000_IG150
 
G_M000_IG153:                ;; offset=0x0F49
       4585F6               test     r14d, r14d
       7430                 je       SHORT G_M000_IG155
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF159C2CCCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       EB36                 jmp      SHORT G_M000_IG157
 
G_M000_IG154:                ;; offset=0x0F76
       4533C0               xor      r8d, r8d
       E98AFEFFFF           jmp      G_M000_IG144
 
G_M000_IG155:                ;; offset=0x0F7E
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF15542CCCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       EB06                 jmp      SHORT G_M000_IG157
 
G_M000_IG156:                ;; offset=0x0FA6
       E805639AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG157:                ;; offset=0x0FAC
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
       E9F6FAFFFF           jmp      G_M000_IG121
 
G_M000_IG158:                ;; offset=0x0FBA
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF156136CCFF         call     [Lokad.Onnx.MathOps:mm(int,int,int,ptr,ptr,ptr)]
       E90AFBFFFF           jmp      G_M000_IG125
 
G_M000_IG159:                ;; offset=0x0FEC
       3BFB                 cmp      edi, ebx
       0F8377070000         jae      G_M000_IG189
       33C0                 xor      eax, eax
       8944B110             mov      dword ptr [rcx+4*rsi+0x10], eax
       413B7808             cmp      edi, dword ptr [r8+0x08]
       0F8367070000         jae      G_M000_IG189
       418B44B010           mov      eax, dword ptr [r8+4*rsi+0x10]
       3BFB                 cmp      edi, ebx
       0F835A070000         jae      G_M000_IG189
       448B5CB210           mov      r11d, dword ptr [rdx+4*rsi+0x10]
       410FAFC3             imul     eax, r11d
       8B9D48FFFFFF         mov      ebx, dword ptr [rbp-0xB8]
       2BD8                 sub      ebx, eax
       899D48FFFFFF         mov      dword ptr [rbp-0xB8], ebx
       413B7908             cmp      edi, dword ptr [r9+0x08]
       0F8339070000         jae      G_M000_IG189
       418BC3               mov      eax, r11d
       410FAF44B110         imul     eax, dword ptr [r9+4*rsi+0x10]
       8B9D44FFFFFF         mov      ebx, dword ptr [rbp-0xBC]
       2BD8                 sub      ebx, eax
       899D44FFFFFF         mov      dword ptr [rbp-0xBC], ebx
       413B7A08             cmp      edi, dword ptr [r10+0x08]
       0F8318070000         jae      G_M000_IG189
       450FAF5CB210         imul     r11d, dword ptr [r10+4*rsi+0x10]
       8BB540FFFFFF         mov      esi, dword ptr [rbp-0xC0]
       412BF3               sub      esi, r11d
       89B540FFFFFF         mov      dword ptr [rbp-0xC0], esi
       FFCF                 dec      edi
       0F89A3FAFFFF         jns      G_M000_IG127
       E946FBFFFF           jmp      G_M000_IG128
 
G_M000_IG160:                ;; offset=0x1075
       81BD14FFFFFF000A0000 cmp      dword ptr [rbp-0xEC], 0xA00
       0F8C7E050000         jl       G_M000_IG183
 
G_M000_IG161:                ;; offset=0x1085
       8BBD94FEFFFF         mov      edi, dword ptr [rbp-0x16C]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF154E35CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4(int,int,int,ptr,ptr,ptr)]
       E9FFF9FFFF           jmp      G_M000_IG122
 
G_M000_IG162:                ;; offset=0x10B7
       448BBD1CFFFFFF       mov      r15d, dword ptr [rbp-0xE4]
       4963F7               movsxd   rsi, r15d
       48BF58B14710677A0000 mov      rdi, 0x7A671047B158
       E8C0F4F57C           call     CORINFO_HELP_NEWARR_1_VC
       488B8D28FEFFFF       mov      rcx, gword ptr [rbp-0x1D8]
       488D7908             lea      rdi, bword ptr [rcx+0x08]
       488BF0               mov      rsi, rax
       E89D4A9AFD           call     CORINFO_HELP_ASSIGN_REF
       4963F7               movsxd   rsi, r15d
       48BF58B14710677A0000 mov      rdi, 0x7A671047B158
       E89BF4F57C           call     CORINFO_HELP_NEWARR_1_VC
       488B8D28FEFFFF       mov      rcx, gword ptr [rbp-0x1D8]
       488D7910             lea      rdi, bword ptr [rcx+0x10]
       488BF0               mov      rsi, rax
       E8784A9AFD           call     CORINFO_HELP_ASSIGN_REF
       4963F7               movsxd   rsi, r15d
       48BF58B14710677A0000 mov      rdi, 0x7A671047B158
       E876F4F57C           call     CORINFO_HELP_NEWARR_1_VC
       488B8D28FEFFFF       mov      rcx, gword ptr [rbp-0x1D8]
       488D7918             lea      rdi, bword ptr [rcx+0x18]
       488BF0               mov      rsi, rax
       E8534A9AFD           call     CORINFO_HELP_ASSIGN_REF
       488B9520FEFFFF       mov      rdx, gword ptr [rbp-0x1E0]
       488D7A10             lea      rdi, bword ptr [rdx+0x10]
       418BF4               mov      esi, r12d
       488B8D28FEFFFF       mov      rcx, gword ptr [rbp-0x1D8]
       4C8B4910             mov      r9, gword ptr [rcx+0x10]
       4C890C24             mov      gword ptr [rsp], r9
       4C8B4918             mov      r9, gword ptr [rcx+0x18]
       4C894C2408           mov      gword ptr [rsp+0x08], r9
       4C8B4908             mov      r9, gword ptr [rcx+0x08]
       488B9518FEFFFF       mov      rdx, gword ptr [rbp-0x1E8]
       488B8D10FEFFFF       mov      rcx, gword ptr [rbp-0x1F0]
       4C8B8508FEFFFF       mov      r8, gword ptr [rbp-0x1F8]
       FF156629CCFF         call     [Lokad.Onnx.Tensor`1[float]:FillBatchOffsets(System.ReadOnlySpan`1[int],int[],int[],int[],int[],int[],int[])]
       48BF70107F11677A0000 mov      rdi, 0x7A67117F1070
       E8FFF2F57C           call     CORINFO_HELP_NEWSFAST
       4C8BF0               mov      r14, rax
       498BFE               mov      rdi, r14
 
G_M000_IG163:                ;; offset=0x1187
       FF156329CCFF         call     [System.Threading.Tasks.ParallelOptions:.ctor():this]
       498BFE               mov      rdi, r14
       8BF3                 mov      esi, ebx
       FF157029CCFF         call     [System.Threading.Tasks.ParallelOptions:set_MaxDegreeOfParallelism(int):this]
       48BF98117F11677A0000 mov      rdi, 0x7A67117F1198
       E8D9F2F57C           call     CORINFO_HELP_NEWSFAST
       488BD8               mov      rbx, rax
       488D7B08             lea      rdi, bword ptr [rbx+0x08]
       488BB528FEFFFF       mov      rsi, gword ptr [rbp-0x1D8]
       E8C6499AFD           call     CORINFO_HELP_ASSIGN_REF
       48BFD89E7511677A0000 mov      rdi, 0x7A6711759ED8
       48897B18             mov      qword ptr [rbx+0x18], rdi
       488DBD20FFFFFF       lea      rdi, [rbp-0xE0]
       4C8BC3               mov      r8, rbx
       418BD7               mov      edx, r15d
       498BCE               mov      rcx, r14
       33F6                 xor      esi, esi
       FF154029CCFF         call     [System.Threading.Tasks.Parallel:For(int,int,System.Threading.Tasks.ParallelOptions,System.Action`1[int]):System.Threading.Tasks.ParallelLoopResult]
       E98C050000           jmp      G_M000_IG190
 
G_M000_IG164:                ;; offset=0x11E5
       81BD14FFFFFF00040000 cmp      dword ptr [rbp-0xEC], 0x400
       0F8CA9FAFFFF         jl       G_M000_IG132
       81BD10FFFFFF00040000 cmp      dword ptr [rbp-0xF0], 0x400
       0F8C99FAFFFF         jl       G_M000_IG132
       48638D14FFFFFF       movsxd   rcx, dword ptr [rbp-0xEC]
       4863BD10FFFFFF       movsxd   rdi, dword ptr [rbp-0xF0]
       480FAFCF             imul     rcx, rdi
       4881F900000004       cmp      rcx, 0x4000000
       0F8F7AFAFFFF         jg       G_M000_IG132
       85DB                 test     ebx, ebx
       0F8472FAFFFF         je       G_M000_IG132
       4585FF               test     r15d, r15d
       0F8469FAFFFF         je       G_M000_IG132
       4C89A5C8FEFFFF       mov      gword ptr [rbp-0x138], r12
       889DF4FEFFFF         mov      byte  ptr [rbp-0x10C], bl
       4488BDF5FEFFFF       mov      byte  ptr [rbp-0x10B], r15b
       488D3C24             lea      rdi, [rsp]
       488DB5C8FEFFFF       lea      rsi, [rbp-0x138]
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
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF159E32CCFF         call     [Lokad.Onnx.Tensor`1[float]:RunIsolatedShortWidePackedRows(int,int,int,ptr,ptr,ptr,Lokad.Onnx.TensorExecutionOptions)]
       E928F8FFFF           jmp      G_M000_IG126
 
G_M000_IG165:                ;; offset=0x12D7
       81BD10FFFFFF00200000 cmp      dword ptr [rbp-0xF0], 0x2000
       0F8CFBF9FFFF         jl       G_M000_IG135
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       BF01000000           mov      edi, 1
       FF15BD32CCFF         call     [Lokad.Onnx.MathOps:mm_m1_kblocked(int,int,int,ptr,ptr,ptr)]
       E9DEF7FFFF           jmp      G_M000_IG125
 
G_M000_IG166:                ;; offset=0x1318
       83BD18FFFFFF40       cmp      dword ptr [rbp-0xE8], 64
       0F8C22FAFFFF         jl       G_M000_IG136
       48638D14FFFFFF       movsxd   rcx, dword ptr [rbp-0xEC]
       4863BD10FFFFFF       movsxd   rdi, dword ptr [rbp-0xF0]
       480FAFCF             imul     rcx, rdi
       4881F900000004       cmp      rcx, 0x4000000
       0F8F03FAFFFF         jg       G_M000_IG136
       4C89A598FEFFFF       mov      gword ptr [rbp-0x168], r12
       889DC4FEFFFF         mov      byte  ptr [rbp-0x13C], bl
       4488BDC5FEFFFF       mov      byte  ptr [rbp-0x13B], r15b
       488D3C24             lea      rdi, [rsp]
       488DB598FEFFFF       lea      rsi, [rbp-0x168]
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
       8BBD14FFFFFF         mov      edi, dword ptr [rbp-0xEC]
       0FAFBD10FFFFFF       imul     edi, dword ptr [rbp-0xF0]
       FF15D931CCFF         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488985C0FDFFFF       mov      gword ptr [rbp-0x240], rax
 
G_M000_IG167:                ;; offset=0x13CE
       488BBDC0FDFFFF       mov      rdi, gword ptr [rbp-0x240]
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
       4883BDC0FDFFFF00     cmp      gword ptr [rbp-0x240], 0
       740D                 je       SHORT G_M000_IG168
       488BBDC0FDFFFF       mov      rdi, gword ptr [rbp-0x240]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       750B                 jne      SHORT G_M000_IG170
 
G_M000_IG168:                ;; offset=0x13F3
       4533E4               xor      r12d, r12d
       EB1E                 jmp      SHORT G_M000_IG171
 
G_M000_IG169:                ;; offset=0x13F8
       E8B35E9AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG170:                ;; offset=0x13FE
       488BBDC0FDFFFF       mov      rdi, gword ptr [rbp-0x240]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       76ED                 jbe      SHORT G_M000_IG169
       4C8BA5C0FDFFFF       mov      r12, gword ptr [rbp-0x240]
       4983C410             add      r12, 16
 
G_M000_IG171:                ;; offset=0x1416
       8BBD14FFFFFF         mov      edi, dword ptr [rbp-0xEC]
       8BB510FFFFFF         mov      esi, dword ptr [rbp-0xF0]
       488B9500FFFFFF       mov      rdx, qword ptr [rbp-0x100]
       498BCC               mov      rcx, r12
       FF15CEABB8FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4D8BC4               mov      r8, r12
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF159D27CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG172:                ;; offset=0x145C
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG173:                ;; offset=0x1465
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB5C0FDFFFF       mov      rsi, gword ptr [rbp-0x240]
       33D2                 xor      edx, edx
       FF157FD4D5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       E9BF010000           jmp      G_M000_IG184
 
G_M000_IG174:                ;; offset=0x1486
       48638D14FFFFFF       movsxd   rcx, dword ptr [rbp-0xEC]
       4863BD10FFFFFF       movsxd   rdi, dword ptr [rbp-0xF0]
       480FAFCF             imul     rcx, rdi
       4881F900000004       cmp      rcx, 0x4000000
       0F8FAFF8FFFF         jg       G_M000_IG137
       4C89A598FEFFFF       mov      gword ptr [rbp-0x168], r12
       889DC4FEFFFF         mov      byte  ptr [rbp-0x13C], bl
       4488BDC5FEFFFF       mov      byte  ptr [rbp-0x13B], r15b
       488D3C24             lea      rdi, [rsp]
       488DB598FEFFFF       lea      rsi, [rbp-0x168]
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
       8BBD14FFFFFF         mov      edi, dword ptr [rbp-0xEC]
       0FAFBD10FFFFFF       imul     edi, dword ptr [rbp-0xF0]
       FF157830CCFF         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488985C8FDFFFF       mov      gword ptr [rbp-0x238], rax
 
G_M000_IG175:                ;; offset=0x152F
       488BBDC8FDFFFF       mov      rdi, gword ptr [rbp-0x238]
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
       4883BDC8FDFFFF00     cmp      gword ptr [rbp-0x238], 0
       740D                 je       SHORT G_M000_IG176
       488BBDC8FDFFFF       mov      rdi, gword ptr [rbp-0x238]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       750B                 jne      SHORT G_M000_IG178
 
G_M000_IG176:                ;; offset=0x1554
       4533F6               xor      r14d, r14d
       EB1E                 jmp      SHORT G_M000_IG179
 
G_M000_IG177:                ;; offset=0x1559
       E8525D9AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG178:                ;; offset=0x155F
       488BBDC8FDFFFF       mov      rdi, gword ptr [rbp-0x238]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       76ED                 jbe      SHORT G_M000_IG177
       4C8BB5C8FDFFFF       mov      r14, gword ptr [rbp-0x238]
       4983C610             add      r14, 16
 
G_M000_IG179:                ;; offset=0x1577
       8BBD14FFFFFF         mov      edi, dword ptr [rbp-0xEC]
       8BB510FFFFFF         mov      esi, dword ptr [rbp-0xF0]
       488B9500FFFFFF       mov      rdx, qword ptr [rbp-0x100]
       498BCE               mov      rcx, r14
       FF156DAAB8FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8BBD94FEFFFF         mov      edi, dword ptr [rbp-0x16C]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4D8BC6               mov      r8, r14
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF155426CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG180:                ;; offset=0x15BD
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG181:                ;; offset=0x15C6
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB5C8FDFFFF       mov      rsi, gword ptr [rbp-0x238]
       33D2                 xor      edx, edx
       FF151ED3D5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       E9CFF4FFFF           jmp      G_M000_IG122
 
G_M000_IG182:                ;; offset=0x15E7
       4963F7               movsxd   rsi, r15d
       48C1E602             shl      rsi, 2
       498BFC               mov      rdi, r12
       49BBA026460F677A0000 mov      r11, 0x7A670F4626A0
       41FF13               call     [r11]Lokad.Onnx.IScratchAccountant:AddScratchBytes(long):this
       E9D3F7FFFF           jmp      G_M000_IG141
 
G_M000_IG183:                ;; offset=0x1603
       81BD10FFFFFF000A0000 cmp      dword ptr [rbp-0xF0], 0xA00
       0F8D72FAFFFF         jge      G_M000_IG161
       8BBD94FEFFFF         mov      edi, dword ptr [rbp-0x16C]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF15A82FCCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4tiled(int,int,int,ptr,ptr,ptr)]
       E971F4FFFF           jmp      G_M000_IG122
 
G_M000_IG184:                ;; offset=0x1645
       4C638D14FFFFFF       movsxd   r9, dword ptr [rbp-0xEC]
       48638D10FFFFFF       movsxd   rcx, dword ptr [rbp-0xF0]
       4C0FAFC9             imul     r9, rcx
       4981F900000004       cmp      r9, 0x4000000
       0F8F7DF4FFFF         jg       G_M000_IG123
       83BD18FFFFFF40       cmp      dword ptr [rbp-0xE8], 64
       7D26                 jge      SHORT G_M000_IG185
       4C638D14FFFFFF       movsxd   r9, dword ptr [rbp-0xEC]
       48638D10FFFFFF       movsxd   rcx, dword ptr [rbp-0xF0]
       4C0FAFC9             imul     r9, rcx
       4981F900000100       cmp      r9, 0x10000
       410F9EC1             setle    r9b
       450FB6C9             movzx    r9, r9b
       E951F4FFFF           jmp      G_M000_IG124
 
G_M000_IG185:                ;; offset=0x1693
       41B901000000         mov      r9d, 1
       E946F4FFFF           jmp      G_M000_IG124
 
G_M000_IG186:                ;; offset=0x169E
       4585C9               test     r9d, r9d
       0F854FF4FFFF         jne      G_M000_IG125
       448B8D94FEFFFF       mov      r9d, dword ptr [rbp-0x16C]
       440FAF8D10FFFFFF     imul     r9d, dword ptr [rbp-0xF0]
       4D63C9               movsxd   r9, r9d
       488B8DF8FEFFFF       mov      rcx, qword ptr [rbp-0x108]
       4E8D0C89             lea      r9, [rcx+4*r9]
       8B8D94FEFFFF         mov      ecx, dword ptr [rbp-0x16C]
       0FAF8D14FFFFFF       imul     ecx, dword ptr [rbp-0xEC]
       4863C9               movsxd   rcx, ecx
       488BB508FFFFFF       mov      rsi, qword ptr [rbp-0xF8]
       488D0C8E             lea      rcx, [rsi+4*rcx]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       BF01000000           mov      edi, 1
       FF151B2FCCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E9F4F3FFFF           jmp      G_M000_IG125
 
G_M000_IG187:                ;; offset=0x1702
       4585FF               test     r15d, r15d
       7432                 je       SHORT G_M000_IG188
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF15E42ECCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E9BDF3FFFF           jmp      G_M000_IG125
 
G_M000_IG188:                ;; offset=0x1739
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF15CA2ECCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized(int,int,int,ptr,ptr,ptr)]
       E98BF3FFFF           jmp      G_M000_IG125
 
G_M000_IG189:                ;; offset=0x176B
       E8405B9AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG190:                ;; offset=0x1771
       48837D9800           cmp      qword ptr [rbp-0x68], 0
       7421                 je       SHORT G_M000_IG192
 
G_M000_IG191:                ;; offset=0x1778
       488D7D98             lea      rdi, bword ptr [rbp-0x68]
       33C0                 xor      eax, eax
       488BD8               mov      rbx, rax
       48871F               xchg     qword ptr [rdi], rbx
       4885DB               test     rbx, rbx
       741F                 je       SHORT G_M000_IG194
       4883E3FE             and      rbx, -2
       488BFB               mov      rdi, rbx
       E83B5FE17C           call     System.Runtime.InteropServices.GCHandle:_InternalFree(nint):bool
       85C0                 test     eax, eax
       7416                 je       SHORT G_M000_IG195
 
G_M000_IG192:                ;; offset=0x1799
       48837D8800           cmp      gword ptr [rbp-0x78], 0
       751A                 jne      SHORT G_M000_IG196
 
G_M000_IG193:                ;; offset=0x17A0
       33FF                 xor      edi, edi
       48897D90             mov      qword ptr [rbp-0x70], rdi
       EB2B                 jmp      SHORT G_M000_IG197
 
G_M000_IG194:                ;; offset=0x17A8
       FF150A1CE1FF         call     [System.ThrowHelper:ThrowInvalidOperationException_HandleIsNotInitialized()]
       CC                   int3     
 
G_M000_IG195:                ;; offset=0x17AF
       488BFB               mov      rdi, rbx
       FF15181CE1FF         call     [System.Runtime.InteropServices.GCHandle:InternalFreeWithGCTransition(nint)]
       EBDF                 jmp      SHORT G_M000_IG192
 
G_M000_IG196:                ;; offset=0x17BA
       488B7D88             mov      rdi, gword ptr [rbp-0x78]
       49BBA826460F677A0000 mov      r11, 0x7A670F4626A8
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33FF                 xor      rdi, rdi
       48897D88             mov      gword ptr [rbp-0x78], rdi
       EBCD                 jmp      SHORT G_M000_IG193
 
G_M000_IG197:                ;; offset=0x17D3
       48837DB000           cmp      qword ptr [rbp-0x50], 0
       7421                 je       SHORT G_M000_IG199
 
G_M000_IG198:                ;; offset=0x17DA
       488D7DB0             lea      rdi, bword ptr [rbp-0x50]
       33C0                 xor      eax, eax
       488BD8               mov      rbx, rax
       48871F               xchg     qword ptr [rdi], rbx
       4885DB               test     rbx, rbx
       741F                 je       SHORT G_M000_IG201
       4883E3FE             and      rbx, -2
       488BFB               mov      rdi, rbx
       E8D95EE17C           call     System.Runtime.InteropServices.GCHandle:_InternalFree(nint):bool
       85C0                 test     eax, eax
       7416                 je       SHORT G_M000_IG202
 
G_M000_IG199:                ;; offset=0x17FB
       48837DA000           cmp      gword ptr [rbp-0x60], 0
       751A                 jne      SHORT G_M000_IG203
 
G_M000_IG200:                ;; offset=0x1802
       33FF                 xor      edi, edi
       48897DA8             mov      qword ptr [rbp-0x58], rdi
       EB2B                 jmp      SHORT G_M000_IG204
 
G_M000_IG201:                ;; offset=0x180A
       FF15A81BE1FF         call     [System.ThrowHelper:ThrowInvalidOperationException_HandleIsNotInitialized()]
       CC                   int3     
 
G_M000_IG202:                ;; offset=0x1811
       488BFB               mov      rdi, rbx
       FF15B61BE1FF         call     [System.Runtime.InteropServices.GCHandle:InternalFreeWithGCTransition(nint)]
       EBDF                 jmp      SHORT G_M000_IG199
 
G_M000_IG203:                ;; offset=0x181C
       488B7DA0             mov      rdi, gword ptr [rbp-0x60]
       49BBB026460F677A0000 mov      r11, 0x7A670F4626B0
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
       EBCD                 jmp      SHORT G_M000_IG200
 
G_M000_IG204:                ;; offset=0x1835
       48837DC800           cmp      qword ptr [rbp-0x38], 0
       7429                 je       SHORT G_M000_IG206
 
G_M000_IG205:                ;; offset=0x183C
       488D7DC8             lea      rdi, bword ptr [rbp-0x38]
       33C0                 xor      eax, eax
       488BD8               mov      rbx, rax
       48871F               xchg     qword ptr [rdi], rbx
       4885DB               test     rbx, rbx
       0F84E60A0000         je       G_M000_IG349
       4883E3FE             and      rbx, -2
       488BFB               mov      rdi, rbx
       E8735EE17C           call     System.Runtime.InteropServices.GCHandle:_InternalFree(nint):bool
       85C0                 test     eax, eax
       0F84D90A0000         je       G_M000_IG350
 
G_M000_IG206:                ;; offset=0x1865
       48837DB800           cmp      gword ptr [rbp-0x48], 0
       0F85DC0A0000         jne      G_M000_IG351
 
G_M000_IG207:                ;; offset=0x1870
       33C0                 xor      eax, eax
       488945C0             mov      qword ptr [rbp-0x40], rax
       E9FBF0FFFF           jmp      G_M000_IG110
 
G_M000_IG208:                ;; offset=0x187B
       4883F820             cmp      rax, 32
       736B                 jae      SHORT G_M000_IG216
 
G_M000_IG209:                ;; offset=0x1881
       4883F810             cmp      rax, 16
       7322                 jae      SHORT G_M000_IG212
 
G_M000_IG210:                ;; offset=0x1887
       4883C0F8             add      rax, -8
       488B11               mov      rdx, qword ptr [rcx]
       482B17               sub      rdx, qword ptr [rdi]
       488B3401             mov      rsi, qword ptr [rcx+rax]
       482B3407             sub      rsi, qword ptr [rdi+rax]
       480BD6               or       rdx, rsi
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
 
G_M000_IG211:                ;; offset=0x18A2
       8BF8                 mov      edi, eax
       E922E9FFFF           jmp      G_M000_IG18
 
G_M000_IG212:                ;; offset=0x18A9
       33C9                 xor      ecx, ecx
       4883C0F0             add      rax, -16
       7519                 jne      SHORT G_M000_IG214
 
G_M000_IG213:                ;; offset=0x18B1
       C5F8100402           vmovups  xmm0, xmmword ptr [rdx+rax]
       62F17D08740C06       vpcmpeqb k1, xmm0, xmmword ptr [rsi+rax]
       C5F898C9             kortestw k1, k1
       736B                 jae      SHORT G_M000_IG220
       E9FEE8FFFF           jmp      G_M000_IG17
 
G_M000_IG214:                ;; offset=0x18C8
       C5F81002             vmovups  xmm0, xmmword ptr [rdx]
       62F37D083E0E04       vpcmpfalseub k1, xmm0, xmmword ptr [rsi]
       C5F898C9             kortestw k1, k1
       7553                 jne      SHORT G_M000_IG220
 
G_M000_IG215:                ;; offset=0x18D9
       4883C110             add      rcx, 16
       483BC1               cmp      rax, rcx
       0F87DC030000         ja       G_M000_IG277
       EBC9                 jmp      SHORT G_M000_IG213
 
G_M000_IG216:                ;; offset=0x18E8
       33FF                 xor      edi, edi
       4883C0E0             add      rax, -32
       7422                 je       SHORT G_M000_IG219
 
G_M000_IG217:                ;; offset=0x18F0
       C5FC1002             vmovups  ymm0, ymmword ptr [rdx]
       62F37D283E0E04       vpcmpfalseub k1, ymm0, ymmword ptr [rsi]
       C4E1F998C9           kortestd k1, k1
       7527                 jne      SHORT G_M000_IG220
 
G_M000_IG218:                ;; offset=0x1902
       4883C720             add      rdi, 32
       483BC7               cmp      rax, rdi
       0F8790030000         ja       G_M000_IG276
 
G_M000_IG219:                ;; offset=0x190F
       C5FC100402           vmovups  ymm0, ymmword ptr [rdx+rax]
       62F17D28740C06       vpcmpeqb k1, ymm0, ymmword ptr [rsi+rax]
       C4E1F998C9           kortestd k1, k1
       0F82A0E8FFFF         jb       G_M000_IG17
 
G_M000_IG220:                ;; offset=0x1926
       33FF                 xor      edi, edi
       E99EE8FFFF           jmp      G_M000_IG18
 
G_M000_IG221:                ;; offset=0x192D
       4883F820             cmp      rax, 32
       736B                 jae      SHORT G_M000_IG229
 
G_M000_IG222:                ;; offset=0x1933
       4883F810             cmp      rax, 16
       7322                 jae      SHORT G_M000_IG225
 
G_M000_IG223:                ;; offset=0x1939
       4883C0F8             add      rax, -8
       488B11               mov      rdx, qword ptr [rcx]
       482B17               sub      rdx, qword ptr [rdi]
       488B3401             mov      rsi, qword ptr [rcx+rax]
       482B3407             sub      rsi, qword ptr [rdi+rax]
       480BD6               or       rdx, rsi
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
 
G_M000_IG224:                ;; offset=0x1954
       8BF8                 mov      edi, eax
       E9CCE9FFFF           jmp      G_M000_IG36
 
G_M000_IG225:                ;; offset=0x195B
       33C9                 xor      ecx, ecx
       4883C0F0             add      rax, -16
       7519                 jne      SHORT G_M000_IG227
 
G_M000_IG226:                ;; offset=0x1963
       C5F8100402           vmovups  xmm0, xmmword ptr [rdx+rax]
       62F17D08740C06       vpcmpeqb k1, xmm0, xmmword ptr [rsi+rax]
       C5F898C9             kortestw k1, k1
       736B                 jae      SHORT G_M000_IG233
       E9A8E9FFFF           jmp      G_M000_IG35
 
G_M000_IG227:                ;; offset=0x197A
       C5F81002             vmovups  xmm0, xmmword ptr [rdx]
       62F37D083E0E04       vpcmpfalseub k1, xmm0, xmmword ptr [rsi]
       C5F898C9             kortestw k1, k1
       7553                 jne      SHORT G_M000_IG233
 
G_M000_IG228:                ;; offset=0x198B
       4883C110             add      rcx, 16
       483BC1               cmp      rax, rcx
       0F87DE040000         ja       G_M000_IG296
       EBC9                 jmp      SHORT G_M000_IG226
 
G_M000_IG229:                ;; offset=0x199A
       33FF                 xor      edi, edi
       4883C0E0             add      rax, -32
       7422                 je       SHORT G_M000_IG232
 
G_M000_IG230:                ;; offset=0x19A2
       C5FC1002             vmovups  ymm0, ymmword ptr [rdx]
       62F37D283E0E04       vpcmpfalseub k1, ymm0, ymmword ptr [rsi]
       C4E1F998C9           kortestd k1, k1
       7527                 jne      SHORT G_M000_IG233
 
G_M000_IG231:                ;; offset=0x19B4
       4883C720             add      rdi, 32
       483BC7               cmp      rax, rdi
       0F8792040000         ja       G_M000_IG295
 
G_M000_IG232:                ;; offset=0x19C1
       C5FC100402           vmovups  ymm0, ymmword ptr [rdx+rax]
       62F17D28740C06       vpcmpeqb k1, ymm0, ymmword ptr [rsi+rax]
       C4E1F998C9           kortestd k1, k1
       0F824AE9FFFF         jb       G_M000_IG35
 
G_M000_IG233:                ;; offset=0x19D8
       33FF                 xor      edi, edi
       E948E9FFFF           jmp      G_M000_IG36
 
G_M000_IG234:                ;; offset=0x19DF
       4883F820             cmp      rax, 32
       736B                 jae      SHORT G_M000_IG242
 
G_M000_IG235:                ;; offset=0x19E5
       4883F810             cmp      rax, 16
       7322                 jae      SHORT G_M000_IG238
 
G_M000_IG236:                ;; offset=0x19EB
       4883C0F8             add      rax, -8
       488B11               mov      rdx, qword ptr [rcx]
       482B17               sub      rdx, qword ptr [rdi]
       488B3401             mov      rsi, qword ptr [rcx+rax]
       482B3407             sub      rsi, qword ptr [rdi+rax]
       480BD6               or       rdx, rsi
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
 
G_M000_IG237:                ;; offset=0x1A06
       8BF8                 mov      edi, eax
       E950EAFFFF           jmp      G_M000_IG51
 
G_M000_IG238:                ;; offset=0x1A0D
       33C9                 xor      ecx, ecx
       4883C0F0             add      rax, -16
       7519                 jne      SHORT G_M000_IG240
 
G_M000_IG239:                ;; offset=0x1A15
       C5F8100402           vmovups  xmm0, xmmword ptr [rdx+rax]
       62F17D08740C06       vpcmpeqb k1, xmm0, xmmword ptr [rsi+rax]
       C5F898C9             kortestw k1, k1
       736B                 jae      SHORT G_M000_IG246
       E92CEAFFFF           jmp      G_M000_IG50
 
G_M000_IG240:                ;; offset=0x1A2C
       C5F81002             vmovups  xmm0, xmmword ptr [rdx]
       62F37D083E0E04       vpcmpfalseub k1, xmm0, xmmword ptr [rsi]
       C5F898C9             kortestw k1, k1
       7553                 jne      SHORT G_M000_IG246
 
G_M000_IG241:                ;; offset=0x1A3D
       4883C110             add      rcx, 16
       483BC1               cmp      rax, rcx
       0F87B3050000         ja       G_M000_IG313
       EBC9                 jmp      SHORT G_M000_IG239
 
G_M000_IG242:                ;; offset=0x1A4C
       33FF                 xor      edi, edi
       4883C0E0             add      rax, -32
       7422                 je       SHORT G_M000_IG245
 
G_M000_IG243:                ;; offset=0x1A54
       C5FC1002             vmovups  ymm0, ymmword ptr [rdx]
       62F37D283E0E04       vpcmpfalseub k1, ymm0, ymmword ptr [rsi]
       C4E1F998C9           kortestd k1, k1
       7527                 jne      SHORT G_M000_IG246
 
G_M000_IG244:                ;; offset=0x1A66
       4883C720             add      rdi, 32
       483BC7               cmp      rax, rdi
       0F8767050000         ja       G_M000_IG312
 
G_M000_IG245:                ;; offset=0x1A73
       C5FC100402           vmovups  ymm0, ymmword ptr [rdx+rax]
       62F17D28740C06       vpcmpeqb k1, ymm0, ymmword ptr [rsi+rax]
       C4E1F998C9           kortestd k1, k1
       0F82CEE9FFFF         jb       G_M000_IG50
 
G_M000_IG246:                ;; offset=0x1A8A
       33FF                 xor      edi, edi
       E9CCE9FFFF           jmp      G_M000_IG51
 
G_M000_IG247:                ;; offset=0x1A91
       8BC6                 mov      eax, esi
       418B448210           mov      eax, dword ptr [r10+4*rax+0x10]
       E9B4EBFFFF           jmp      G_M000_IG70
 
G_M000_IG248:                ;; offset=0x1A9D
       448B8D48FEFFFF       mov      r9d, dword ptr [rbp-0x1B8]
       458D41FE             lea      r8d, [r9-0x02]
       413BF0               cmp      esi, r8d
       7D1C                 jge      SHORT G_M000_IG250
 
G_M000_IG249:                ;; offset=0x1AAD
       413BF1               cmp      esi, r9d
       0F8394080000         jae      G_M000_IG352
       8BC6                 mov      eax, esi
       488B9588FDFFFF       mov      rdx, bword ptr [rbp-0x278]
       833C8201             cmp      dword ptr [rdx+4*rax], 1
       0F8594060000         jne      G_M000_IG329
 
G_M000_IG250:                ;; offset=0x1AC9
       33C0                 xor      eax, eax
 
G_M000_IG251:                ;; offset=0x1ACB
       3B7108               cmp      esi, dword ptr [rcx+0x08]
       0F8376080000         jae      G_M000_IG352
       448BC6               mov      r8d, esi
       4289448110           mov      dword ptr [rcx+4*r8+0x10], eax
       FFC6                 inc      esi
       3BF7                 cmp      esi, edi
       7CBB                 jl       SHORT G_M000_IG248
       E978EBFFFF           jmp      G_M000_IG71
 
G_M000_IG252:                ;; offset=0x1AE7
       8BCE                 mov      ecx, esi
       418B4C8910           mov      ecx, dword ptr [r9+4*rcx+0x10]
       E942ECFFFF           jmp      G_M000_IG84
 
G_M000_IG253:                ;; offset=0x1AF3
       448B9538FEFFFF       mov      r10d, dword ptr [rbp-0x1C8]
       418D52FE             lea      edx, [r10-0x02]
       3BF2                 cmp      esi, edx
       7D1D                 jge      SHORT G_M000_IG255
 
G_M000_IG254:                ;; offset=0x1B02
       413BF2               cmp      esi, r10d
       0F833F080000         jae      G_M000_IG352
       8BCE                 mov      ecx, esi
       4C8B8578FDFFFF       mov      r8, bword ptr [rbp-0x288]
       41833C8801           cmp      dword ptr [r8+4*rcx], 1
       0F85D5060000         jne      G_M000_IG336
 
G_M000_IG255:                ;; offset=0x1B1F
       33C9                 xor      ecx, ecx
 
G_M000_IG256:                ;; offset=0x1B21
       3B7708               cmp      esi, dword ptr [rdi+0x08]
       0F8320080000         jae      G_M000_IG352
       8BD6                 mov      edx, esi
       894C9710             mov      dword ptr [rdi+4*rdx+0x10], ecx
       FFC6                 inc      esi
       3BF0                 cmp      esi, eax
       7CBD                 jl       SHORT G_M000_IG253
       E908ECFFFF           jmp      G_M000_IG85
 
G_M000_IG257:                ;; offset=0x1B3B
       8BC8                 mov      ecx, eax
       8B4C8F10             mov      ecx, dword ptr [rdi+4*rcx+0x10]
       E9A7ECFFFF           jmp      G_M000_IG95
 
G_M000_IG258:                ;; offset=0x1B46
       448B9530FEFFFF       mov      r10d, dword ptr [rbp-0x1D0]
       418D72FE             lea      esi, [r10-0x02]
       3BC6                 cmp      eax, esi
       7D1D                 jge      SHORT G_M000_IG260
 
G_M000_IG259:                ;; offset=0x1B55
       413BC2               cmp      eax, r10d
       0F83EC070000         jae      G_M000_IG352
       8BC8                 mov      ecx, eax
       4C8B8D70FDFFFF       mov      r9, bword ptr [rbp-0x290]
       41833C8901           cmp      dword ptr [r9+4*rcx], 1
       0F85B0060000         jne      G_M000_IG341
 
G_M000_IG260:                ;; offset=0x1B72
       33C9                 xor      ecx, ecx
 
G_M000_IG261:                ;; offset=0x1B74
       413B4008             cmp      eax, dword ptr [r8+0x08]
       0F83CC070000         jae      G_M000_IG352
       8BF0                 mov      esi, eax
       41894CB010           mov      dword ptr [r8+4*rsi+0x10], ecx
       FFC0                 inc      eax
       3BC2                 cmp      eax, edx
       7CBB                 jl       SHORT G_M000_IG258
       E96BECFFFF           jmp      G_M000_IG96
 
G_M000_IG262:                ;; offset=0x1B90
       498BF7               mov      rsi, r15
       48BF301A7C11677A0000 mov      rdi, 0x7A67117C1A30
       E87E87EAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       488BF8               mov      rdi, rax
       E908E5FFFF           jmp      G_M000_IG04
 
G_M000_IG263:                ;; offset=0x1BAA
       FF15A01FCCFF         call     [Lokad.Onnx.Tensor`1[float]:HasDenseMatrixCore(Lokad.Onnx.BroadcastedTensor`1[float]):bool]
       85C0                 test     eax, eax
       0F8403E5FFFF         je       G_M000_IG05
       E929E6FFFF           jmp      G_M000_IG20
 
G_M000_IG264:                ;; offset=0x1BBD
       498BF7               mov      rsi, r15
       E85B87EAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BE0               mov      r12, rax
       E90AE5FFFF           jmp      G_M000_IG07
 
G_M000_IG265:                ;; offset=0x1BCD
       33C9                 xor      rcx, rcx
       48898DA8FDFFFF       mov      bword ptr [rbp-0x258], rcx
       4533C0               xor      r8d, r8d
       44898560FEFFFF       mov      dword ptr [rbp-0x1A0], r8d
       488B8DA8FDFFFF       mov      rcx, bword ptr [rbp-0x258]
       448B8560FEFFFF       mov      r8d, dword ptr [rbp-0x1A0]
       E90FE5FFFF           jmp      G_M000_IG08
 
G_M000_IG266:                ;; offset=0x1BF3
       33F6                 xor      rsi, rsi
       33D2                 xor      edx, edx
       48898DA8FDFFFF       mov      bword ptr [rbp-0x258], rcx
       44898560FEFFFF       mov      dword ptr [rbp-0x1A0], r8d
       E91BE5FFFF           jmp      G_M000_IG09
 
G_M000_IG267:                ;; offset=0x1C0A
       33FF                 xor      rdi, rdi
       33C0                 xor      eax, eax
       E92FE5FFFF           jmp      G_M000_IG10
 
G_M000_IG268:                ;; offset=0x1C13
       4883F804             cmp      rax, 4
       7332                 jae      SHORT G_M000_IG274
 
G_M000_IG269:                ;; offset=0x1C19
       33D2                 xor      edx, edx
       488BF0               mov      rsi, rax
       4883E602             and      rsi, 2
       740A                 je       SHORT G_M000_IG271
 
G_M000_IG270:                ;; offset=0x1C24
       0FB711               movzx    rdx, word  ptr [rcx]
       440FB707             movzx    r8, word  ptr [rdi]
       412BD0               sub      edx, r8d
 
G_M000_IG271:                ;; offset=0x1C2E
       A801                 test     al, 1
       740C                 je       SHORT G_M000_IG273
 
G_M000_IG272:                ;; offset=0x1C32
       0FB60431             movzx    rax, byte  ptr [rcx+rsi]
       0FB60C37             movzx    rcx, byte  ptr [rdi+rsi]
       2BC1                 sub      eax, ecx
       0BD0                 or       edx, eax
 
G_M000_IG273:                ;; offset=0x1C3E
       85D2                 test     edx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E957FCFFFF           jmp      G_M000_IG211
 
G_M000_IG274:                ;; offset=0x1C4B
       4883C0FC             add      rax, -4
       8B11                 mov      edx, dword ptr [rcx]
       2B17                 sub      edx, dword ptr [rdi]
       8B0C01               mov      ecx, dword ptr [rcx+rax]
       2B0C07               sub      ecx, dword ptr [rdi+rax]
       0BCA                 or       ecx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E93CFCFFFF           jmp      G_M000_IG211
 
G_M000_IG275:                ;; offset=0x1C66
       62F17C4810040A       vmovups  zmm0, zmmword ptr [rdx+rcx]
       62F37D483E0C0E04     vpcmpfalseub k1, zmm0, zmmword ptr [rsi+rcx]
       C4E1F898C9           kortestq k1, k1
       0F85A6FCFFFF         jne      G_M000_IG220
       E918E5FFFF           jmp      G_M000_IG15
 
G_M000_IG276:                ;; offset=0x1C85
       C5FC10043A           vmovups  ymm0, ymmword ptr [rdx+rdi]
       62F37D283E0C3E04     vpcmpfalseub k1, ymm0, ymmword ptr [rsi+rdi]
       C4E1F998C9           kortestd k1, k1
       0F8589FCFFFF         jne      G_M000_IG220
       E960FCFFFF           jmp      G_M000_IG218
 
G_M000_IG277:                ;; offset=0x1CA2
       C5F810040A           vmovups  xmm0, xmmword ptr [rdx+rcx]
       62F37D083E0C0E04     vpcmpfalseub k1, xmm0, xmmword ptr [rsi+rcx]
       C5F898C9             kortestw k1, k1
       0F856DFCFFFF         jne      G_M000_IG220
       E91BFCFFFF           jmp      G_M000_IG215
 
G_M000_IG278:                ;; offset=0x1CBE
       33FF                 xor      edi, edi
       E906E5FFFF           jmp      G_M000_IG18
 
G_M000_IG279:                ;; offset=0x1CC5
       48BF28CD4810677A0000 mov      rdi, 0x7A671048CD28
       E8ACE7F57C           call     CORINFO_HELP_NEWSFAST
       4C8BF8               mov      r15, rax
       BF53E60000           mov      edi, 0xE653
       48BE087F5410677A0000 mov      rsi, 0x7A6710547F08
       FF155C8261FF         call     [CORINFO_HELP_STRCNS]
       4C8BE8               mov      r13, rax
       BF1FE90000           mov      edi, 0xE91F
       48BE087F5410677A0000 mov      rsi, 0x7A6710547F08
       FF15448261FF         call     [CORINFO_HELP_STRCNS]
       488BF0               mov      rsi, rax
       498BFD               mov      rdi, r13
       FF15981A9BFE         call     [System.String:Concat(System.String,System.String):System.String]
       488BF0               mov      rsi, rax
       498BFF               mov      rdi, r15
       FF158C8261FF         call     [System.ArgumentException:.ctor(System.String):this]
       498BFF               mov      rdi, r15
       E8FC13E17C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG280:                ;; offset=0x1D25
       498BFF               mov      rdi, r15
       498B07               mov      rax, qword ptr [r15]
       488B4078             mov      rax, qword ptr [rax+0x78]
       FF5010               call     [rax+0x10]Lokad.Onnx.Tensor`1[float]:ToDenseTensor():Lokad.Onnx.DenseTensor`1[float]:this
       488BF8               mov      rdi, rax
       498BF5               mov      rsi, r13
       FF158A0CCCFF         call     [Lokad.Onnx.Tensor`1[float]:CountedCopy[float](Lokad.Onnx.DenseTensor`1[float],Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE0               mov      r12, rax
       E99DE4FFFF           jmp      G_M000_IG19
 
G_M000_IG281:                ;; offset=0x1D46
       488BF3               mov      rsi, rbx
       48BF301A7C11677A0000 mov      rdi, 0x7A67117C1A30
       E8C885EAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       488BF8               mov      rdi, rax
       E9AEE4FFFF           jmp      G_M000_IG22
 
G_M000_IG282:                ;; offset=0x1D60
       FF15EA1DCCFF         call     [Lokad.Onnx.Tensor`1[float]:HasDenseMatrixCore(Lokad.Onnx.BroadcastedTensor`1[float]):bool]
       85C0                 test     eax, eax
       0F84A9E4FFFF         je       G_M000_IG23
       E9CFE5FFFF           jmp      G_M000_IG38
 
G_M000_IG283:                ;; offset=0x1D73
       488BF3               mov      rsi, rbx
       E8A585EAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BE0               mov      r12, rax
       E9B0E4FFFF           jmp      G_M000_IG25
 
G_M000_IG284:                ;; offset=0x1D83
       33C9                 xor      rcx, rcx
       48898DA0FDFFFF       mov      bword ptr [rbp-0x260], rcx
       4533C0               xor      r8d, r8d
       4489855CFEFFFF       mov      dword ptr [rbp-0x1A4], r8d
       488B8DA0FDFFFF       mov      rcx, bword ptr [rbp-0x260]
       448B855CFEFFFF       mov      r8d, dword ptr [rbp-0x1A4]
       E9B5E4FFFF           jmp      G_M000_IG26
 
G_M000_IG285:                ;; offset=0x1DA9
       33F6                 xor      rsi, rsi
       33D2                 xor      edx, edx
       48898DA0FDFFFF       mov      bword ptr [rbp-0x260], rcx
       4489855CFEFFFF       mov      dword ptr [rbp-0x1A4], r8d
       E9C1E4FFFF           jmp      G_M000_IG27
 
G_M000_IG286:                ;; offset=0x1DC0
       33FF                 xor      rdi, rdi
       33C0                 xor      eax, eax
       E9D5E4FFFF           jmp      G_M000_IG28
 
G_M000_IG287:                ;; offset=0x1DC9
       4883F804             cmp      rax, 4
       7332                 jae      SHORT G_M000_IG293
 
G_M000_IG288:                ;; offset=0x1DCF
       33D2                 xor      edx, edx
       488BF0               mov      rsi, rax
       4883E602             and      rsi, 2
       740A                 je       SHORT G_M000_IG290
 
G_M000_IG289:                ;; offset=0x1DDA
       0FB711               movzx    rdx, word  ptr [rcx]
       440FB707             movzx    r8, word  ptr [rdi]
       412BD0               sub      edx, r8d
 
G_M000_IG290:                ;; offset=0x1DE4
       A801                 test     al, 1
       740C                 je       SHORT G_M000_IG292
 
G_M000_IG291:                ;; offset=0x1DE8
       0FB60431             movzx    rax, byte  ptr [rcx+rsi]
       0FB60C37             movzx    rcx, byte  ptr [rdi+rsi]
       2BC1                 sub      eax, ecx
       0BD0                 or       edx, eax
 
G_M000_IG292:                ;; offset=0x1DF4
       85D2                 test     edx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E953FBFFFF           jmp      G_M000_IG224
 
G_M000_IG293:                ;; offset=0x1E01
       4883C0FC             add      rax, -4
       8B11                 mov      edx, dword ptr [rcx]
       2B17                 sub      edx, dword ptr [rdi]
       8B0C01               mov      ecx, dword ptr [rcx+rax]
       2B0C07               sub      ecx, dword ptr [rdi+rax]
       0BCA                 or       ecx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E938FBFFFF           jmp      G_M000_IG224
 
G_M000_IG294:                ;; offset=0x1E1C
       62F17C4810040A       vmovups  zmm0, zmmword ptr [rdx+rcx]
       62F37D483E0C0E04     vpcmpfalseub k1, zmm0, zmmword ptr [rsi+rcx]
       C4E1F898C9           kortestq k1, k1
       0F85A2FBFFFF         jne      G_M000_IG233
       E9BEE4FFFF           jmp      G_M000_IG33
 
G_M000_IG295:                ;; offset=0x1E3B
       C5FC10043A           vmovups  ymm0, ymmword ptr [rdx+rdi]
       62F37D283E0C3E04     vpcmpfalseub k1, ymm0, ymmword ptr [rsi+rdi]
       C4E1F998C9           kortestd k1, k1
       0F8585FBFFFF         jne      G_M000_IG233
       E95CFBFFFF           jmp      G_M000_IG231
 
G_M000_IG296:                ;; offset=0x1E58
       C5F810040A           vmovups  xmm0, xmmword ptr [rdx+rcx]
       62F37D083E0C0E04     vpcmpfalseub k1, xmm0, xmmword ptr [rsi+rcx]
       C5F898C9             kortestw k1, k1
       0F8569FBFFFF         jne      G_M000_IG233
       E917FBFFFF           jmp      G_M000_IG228
 
G_M000_IG297:                ;; offset=0x1E74
       33FF                 xor      edi, edi
       E9ACE4FFFF           jmp      G_M000_IG36
 
G_M000_IG298:                ;; offset=0x1E7B
       48BF28CD4810677A0000 mov      rdi, 0x7A671048CD28
       E8F6E5F57C           call     CORINFO_HELP_NEWSFAST
       488BD8               mov      rbx, rax
       BF59E60000           mov      edi, 0xE659
       48BE087F5410677A0000 mov      rsi, 0x7A6710547F08
       FF15A68061FF         call     [CORINFO_HELP_STRCNS]
       4C8BE8               mov      r13, rax
       BF1FE90000           mov      edi, 0xE91F
       48BE087F5410677A0000 mov      rsi, 0x7A6710547F08
       FF158E8061FF         call     [CORINFO_HELP_STRCNS]
       488BF0               mov      rsi, rax
       498BFD               mov      rdi, r13
       FF15E2189BFE         call     [System.String:Concat(System.String,System.String):System.String]
       488BF0               mov      rsi, rax
       488BFB               mov      rdi, rbx
       FF15D68061FF         call     [System.ArgumentException:.ctor(System.String):this]
       488BFB               mov      rdi, rbx
       E84612E17C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG299:                ;; offset=0x1EDB
       488BFB               mov      rdi, rbx
       488B03               mov      rax, qword ptr [rbx]
       488B4078             mov      rax, qword ptr [rax+0x78]
       FF5010               call     [rax+0x10]Lokad.Onnx.Tensor`1[float]:ToDenseTensor():Lokad.Onnx.DenseTensor`1[float]:this
       488BF8               mov      rdi, rax
       498BF5               mov      rsi, r13
       FF15D40ACCFF         call     [Lokad.Onnx.Tensor`1[float]:CountedCopy[float](Lokad.Onnx.DenseTensor`1[float],Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE0               mov      r12, rax
       E943E4FFFF           jmp      G_M000_IG37
 
G_M000_IG300:                ;; offset=0x1EFC
       498BF6               mov      rsi, r14
       E81C84EAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BE0               mov      r12, rax
       E95DE4FFFF           jmp      G_M000_IG40
 
G_M000_IG301:                ;; offset=0x1F0C
       33C9                 xor      rcx, rcx
       48898D98FDFFFF       mov      bword ptr [rbp-0x268], rcx
       4533C0               xor      r8d, r8d
       44898558FEFFFF       mov      dword ptr [rbp-0x1A8], r8d
       488B8D98FDFFFF       mov      rcx, bword ptr [rbp-0x268]
       448B8558FEFFFF       mov      r8d, dword ptr [rbp-0x1A8]
       E962E4FFFF           jmp      G_M000_IG41
 
G_M000_IG302:                ;; offset=0x1F32
       33F6                 xor      rsi, rsi
       33D2                 xor      edx, edx
       48898D98FDFFFF       mov      bword ptr [rbp-0x268], rcx
       44898558FEFFFF       mov      dword ptr [rbp-0x1A8], r8d
       E96EE4FFFF           jmp      G_M000_IG42
 
G_M000_IG303:                ;; offset=0x1F49
       33FF                 xor      rdi, rdi
       33C0                 xor      eax, eax
       E982E4FFFF           jmp      G_M000_IG43
 
G_M000_IG304:                ;; offset=0x1F52
       4883F804             cmp      rax, 4
       7332                 jae      SHORT G_M000_IG310
 
G_M000_IG305:                ;; offset=0x1F58
       33D2                 xor      edx, edx
       488BF0               mov      rsi, rax
       4883E602             and      rsi, 2
       740A                 je       SHORT G_M000_IG307
 
G_M000_IG306:                ;; offset=0x1F63
       0FB711               movzx    rdx, word  ptr [rcx]
       440FB707             movzx    r8, word  ptr [rdi]
       412BD0               sub      edx, r8d
 
G_M000_IG307:                ;; offset=0x1F6D
       A801                 test     al, 1
       740C                 je       SHORT G_M000_IG309
 
G_M000_IG308:                ;; offset=0x1F71
       0FB60431             movzx    rax, byte  ptr [rcx+rsi]
       0FB60C37             movzx    rcx, byte  ptr [rdi+rsi]
       2BC1                 sub      eax, ecx
       0BD0                 or       edx, eax
 
G_M000_IG309:                ;; offset=0x1F7D
       85D2                 test     edx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E97CFAFFFF           jmp      G_M000_IG237
 
G_M000_IG310:                ;; offset=0x1F8A
       4883C0FC             add      rax, -4
       8B11                 mov      edx, dword ptr [rcx]
       2B17                 sub      edx, dword ptr [rdi]
       8B0C01               mov      ecx, dword ptr [rcx+rax]
       2B0C07               sub      ecx, dword ptr [rdi+rax]
       0BCA                 or       ecx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E961FAFFFF           jmp      G_M000_IG237
 
G_M000_IG311:                ;; offset=0x1FA5
       62F17C4810040A       vmovups  zmm0, zmmword ptr [rdx+rcx]
       62F37D483E0C0E04     vpcmpfalseub k1, zmm0, zmmword ptr [rsi+rcx]
       C4E1F898C9           kortestq k1, k1
       0F85CBFAFFFF         jne      G_M000_IG246
       E96BE4FFFF           jmp      G_M000_IG48
 
G_M000_IG312:                ;; offset=0x1FC4
       C5FC10043A           vmovups  ymm0, ymmword ptr [rdx+rdi]
       62F37D283E0C3E04     vpcmpfalseub k1, ymm0, ymmword ptr [rsi+rdi]
       C4E1F998C9           kortestd k1, k1
       0F85AEFAFFFF         jne      G_M000_IG246
       E985FAFFFF           jmp      G_M000_IG244
 
G_M000_IG313:                ;; offset=0x1FE1
       C5F810040A           vmovups  xmm0, xmmword ptr [rdx+rcx]
       62F37D083E0C0E04     vpcmpfalseub k1, xmm0, xmmword ptr [rsi+rcx]
       C5F898C9             kortestw k1, k1
       0F8592FAFFFF         jne      G_M000_IG246
       E940FAFFFF           jmp      G_M000_IG241
 
G_M000_IG314:                ;; offset=0x1FFD
       33FF                 xor      edi, edi
       E959E4FFFF           jmp      G_M000_IG51
 
G_M000_IG315:                ;; offset=0x2004
       48BF28CD4810677A0000 mov      rdi, 0x7A671048CD28
       E86DE4F57C           call     CORINFO_HELP_NEWSFAST
       4C8BF0               mov      r14, rax
       BF3AE80000           mov      edi, 0xE83A
       48BE087F5410677A0000 mov      rsi, 0x7A6710547F08
       FF151D7F61FF         call     [CORINFO_HELP_STRCNS]
       4C8BE8               mov      r13, rax
       BF1FE90000           mov      edi, 0xE91F
       48BE087F5410677A0000 mov      rsi, 0x7A6710547F08
       FF15057F61FF         call     [CORINFO_HELP_STRCNS]
       488BF0               mov      rsi, rax
       498BFD               mov      rdi, r13
       FF1559179BFE         call     [System.String:Concat(System.String,System.String):System.String]
       488BF0               mov      rsi, rax
       498BFE               mov      rdi, r14
       FF154D7F61FF         call     [System.ArgumentException:.ctor(System.String):this]
       498BFE               mov      rdi, r14
       E8BD10E17C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG316:                ;; offset=0x2064
       498BFE               mov      rdi, r14
       498B06               mov      rax, qword ptr [r14]
       488B4078             mov      rax, qword ptr [rax+0x78]
       FF5010               call     [rax+0x10]Lokad.Onnx.Tensor`1[float]:ToDenseTensor():Lokad.Onnx.DenseTensor`1[float]:this
       488BF8               mov      rdi, rax
       498BF5               mov      rsi, r13
       FF154B09CCFF         call     [Lokad.Onnx.Tensor`1[float]:CountedCopy[float](Lokad.Onnx.DenseTensor`1[float],Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE0               mov      r12, rax
       E9F0E3FFFF           jmp      G_M000_IG52
 
G_M000_IG317:                ;; offset=0x2085
       33FF                 xor      rdi, rdi
       33F6                 xor      esi, esi
       E9FEE3FFFF           jmp      G_M000_IG53
 
G_M000_IG318:                ;; offset=0x208E
       FF15849614FF         call     [System.ThrowHelper:ThrowArgumentOutOfRangeException()]
       CC                   int3     
 
G_M000_IG319:                ;; offset=0x2095
       49BD80A8800C677A0000 mov      r13, 0x7A670C80A880
       E94BE4FFFF           jmp      G_M000_IG54
 
G_M000_IG320:                ;; offset=0x20A4
       33C0                 xor      rax, rax
       33FF                 xor      edi, edi
       E960E4FFFF           jmp      G_M000_IG55
 
G_M000_IG321:                ;; offset=0x20AD
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E986E4FFFF           jmp      G_M000_IG56
 
G_M000_IG322:                ;; offset=0x20B6
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E9A4E4FFFF           jmp      G_M000_IG57
 
G_M000_IG323:                ;; offset=0x20BF
       33D2                 xor      rdx, rdx
       48899590FDFFFF       mov      bword ptr [rbp-0x270], rdx
       4533C0               xor      r8d, r8d
       44898550FEFFFF       mov      dword ptr [rbp-0x1B0], r8d
       488B9590FDFFFF       mov      rdx, bword ptr [rbp-0x270]
       448B8550FEFFFF       mov      r8d, dword ptr [rbp-0x1B0]
       E9B4E4FFFF           jmp      G_M000_IG58
 
G_M000_IG324:                ;; offset=0x20E5
       48899590FDFFFF       mov      bword ptr [rbp-0x270], rdx
       44898550FEFFFF       mov      dword ptr [rbp-0x1B0], r8d
       498BF7               mov      rsi, r15
       48BF301A7C11677A0000 mov      rdi, 0x7A67117C1A30
       E81B82EAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BC8               mov      r9, rax
       488B9590FDFFFF       mov      rdx, bword ptr [rbp-0x270]
       448B8550FEFFFF       mov      r8d, dword ptr [rbp-0x1B0]
       E997E4FFFF           jmp      G_M000_IG59
 
G_M000_IG325:                ;; offset=0x211B
       4983795000           cmp      gword ptr [r9+0x50], 0
       0F8495E4FFFF         je       G_M000_IG60
       4D8B5150             mov      r10, gword ptr [r9+0x50]
       4C8995E8FDFFFF       mov      gword ptr [rbp-0x218], r10
       E990E4FFFF           jmp      G_M000_IG61
 
G_M000_IG326:                ;; offset=0x2136
       E962F9FFFF           jmp      G_M000_IG248
 
G_M000_IG327:                ;; offset=0x213B
       E95DF9FFFF           jmp      G_M000_IG248
 
G_M000_IG328:                ;; offset=0x2140
       E958F9FFFF           jmp      G_M000_IG248
 
G_M000_IG329:                ;; offset=0x2145
       413B7208             cmp      esi, dword ptr [r10+0x08]
       0F83E3010000         jae      G_M000_IG352
       8BC6                 mov      eax, esi
       418B448210           mov      eax, dword ptr [r10+4*rax+0x10]
       E970F9FFFF           jmp      G_M000_IG251
 
G_M000_IG330:                ;; offset=0x215B
       33C9                 xor      rcx, rcx
       48898D80FDFFFF       mov      bword ptr [rbp-0x280], rcx
       33D2                 xor      edx, edx
       899540FEFFFF         mov      dword ptr [rbp-0x1C0], edx
       488B8D80FDFFFF       mov      rcx, bword ptr [rbp-0x280]
       8B9540FEFFFF         mov      edx, dword ptr [rbp-0x1C0]
       E903E5FFFF           jmp      G_M000_IG72
 
G_M000_IG331:                ;; offset=0x217E
       48898D80FDFFFF       mov      bword ptr [rbp-0x280], rcx
       899540FEFFFF         mov      dword ptr [rbp-0x1C0], edx
       488BF3               mov      rsi, rbx
       48BF301A7C11677A0000 mov      rdi, 0x7A67117C1A30
       E88381EAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BC0               mov      r8, rax
       488B8D80FDFFFF       mov      rcx, bword ptr [rbp-0x280]
       8B9540FEFFFF         mov      edx, dword ptr [rbp-0x1C0]
       E9E8E4FFFF           jmp      G_M000_IG73
 
G_M000_IG332:                ;; offset=0x21B2
       4983785000           cmp      gword ptr [r8+0x50], 0
       0F84E6E4FFFF         je       G_M000_IG74
       4D8B4850             mov      r9, gword ptr [r8+0x50]
       4C898DE0FDFFFF       mov      gword ptr [rbp-0x220], r9
       E9E1E4FFFF           jmp      G_M000_IG75
 
G_M000_IG333:                ;; offset=0x21CD
       E921F9FFFF           jmp      G_M000_IG253
 
G_M000_IG334:                ;; offset=0x21D2
       E91CF9FFFF           jmp      G_M000_IG253
 
G_M000_IG335:                ;; offset=0x21D7
       E917F9FFFF           jmp      G_M000_IG253
 
G_M000_IG336:                ;; offset=0x21DC
       413B7108             cmp      esi, dword ptr [r9+0x08]
       0F834C010000         jae      G_M000_IG352
       8BCE                 mov      ecx, esi
       418B4C8910           mov      ecx, dword ptr [r9+4*rcx+0x10]
       E92FF9FFFF           jmp      G_M000_IG256
 
G_M000_IG337:                ;; offset=0x21F2
       33C9                 xor      rcx, rcx
       33D2                 xor      edx, edx
       E966E5FFFF           jmp      G_M000_IG86
 
G_M000_IG338:                ;; offset=0x21FB
       E946F9FFFF           jmp      G_M000_IG258
 
G_M000_IG339:                ;; offset=0x2200
       E941F9FFFF           jmp      G_M000_IG258
 
G_M000_IG340:                ;; offset=0x2205
       E93CF9FFFF           jmp      G_M000_IG258
 
G_M000_IG341:                ;; offset=0x220A
       3B4708               cmp      eax, dword ptr [rdi+0x08]
       0F831F010000         jae      G_M000_IG352
       8BF0                 mov      esi, eax
       8B4CB710             mov      ecx, dword ptr [rdi+4*rsi+0x10]
       E956F9FFFF           jmp      G_M000_IG261
 
G_M000_IG342:                ;; offset=0x221E
       837DD402             cmp      dword ptr [rbp-0x2C], 2
       0F8C19E6FFFF         jl       G_M000_IG100
       8B4870               mov      ecx, dword ptr [rax+0x70]
       448B45D4             mov      r8d, dword ptr [rbp-0x2C]
       413BC8               cmp      ecx, r8d
       410F4FC8             cmovg    ecx, r8d
       898D38FFFFFF         mov      dword ptr [rbp-0xC8], ecx
       E90AE6FFFF           jmp      G_M000_IG101
 
G_M000_IG343:                ;; offset=0x2241
       BA56555555           mov      edx, 0x55555556
       8BC2                 mov      eax, edx
       41F7ED               imul     edx:eax, r13d
       8BC2                 mov      eax, edx
       C1E81F               shr      eax, 31
       03C2                 add      eax, edx
       8D0440               lea      eax, [rax+2*rax]
       442BE8               sub      r13d, eax
       0F852EE7FFFF         jne      G_M000_IG112
       E914E6FFFF           jmp      G_M000_IG102
 
G_M000_IG344:                ;; offset=0x2263
       33C0                 xor      eax, eax
       E938E6FFFF           jmp      G_M000_IG104
 
G_M000_IG345:                ;; offset=0x226A
       33C9                 xor      rcx, rcx
       33C0                 xor      eax, eax
       E94BE6FFFF           jmp      G_M000_IG105
 
G_M000_IG346:                ;; offset=0x2273
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E95CE6FFFF           jmp      G_M000_IG106
 
G_M000_IG347:                ;; offset=0x227C
       48BA882D80295F7A0000 mov      rdx, 0x7A5F29802D88
       488B12               mov      rdx, gword ptr [rdx]
       4885D2               test     rdx, rdx
       7556                 jne      SHORT G_M000_IG348
       48BFE82F5811677A0000 mov      rdi, 0x7A6711582FE8
       E8E3E1F57C           call     CORINFO_HELP_NEWSFAST
       488BD0               mov      rdx, rax
       488995F8FDFFFF       mov      gword ptr [rbp-0x208], rdx
       48BE702B80295F7A0000 mov      rsi, 0x7A5F29802B70
       488B36               mov      rsi, gword ptr [rsi]
       488BFA               mov      rdi, rdx
       48BAC09E7511677A0000 mov      rdx, 0x7A6711759EC0
       FF1581A69AFE         call     [System.MulticastDelegate:CtorClosed(System.Object,nint):this]
       48BF882D80295F7A0000 mov      rdi, 0x7A5F29802D88
       488BB5F8FDFFFF       mov      rsi, gword ptr [rbp-0x208]
       E8A3389AFD           call     CORINFO_HELP_ASSIGN_REF
       488B95F8FDFFFF       mov      rdx, gword ptr [rbp-0x208]
 
G_M000_IG348:                ;; offset=0x22E4
       488BBD10FEFFFF       mov      rdi, gword ptr [rbp-0x1F0]
       488BF2               mov      rsi, rdx
       FF15C47EB8FF         call     [System.Linq.Enumerable:All[int](System.Collections.Generic.IEnumerable`1[int],System.Func`2[int,bool]):bool]
       85C0                 test     eax, eax
       0F8495E6FFFF         je       G_M000_IG113
       E923E6FFFF           jmp      G_M000_IG109
 
G_M000_IG349:                ;; offset=0x2301
       FF15B110E1FF         call     [System.ThrowHelper:ThrowInvalidOperationException_HandleIsNotInitialized()]
       CC                   int3     
 
G_M000_IG350:                ;; offset=0x2308
       488BFB               mov      rdi, rbx
       FF15BF10E1FF         call     [System.Runtime.InteropServices.GCHandle:InternalFreeWithGCTransition(nint)]
       E94FF5FFFF           jmp      G_M000_IG206
 
G_M000_IG351:                ;; offset=0x2316
       488B7DB8             mov      rdi, gword ptr [rbp-0x48]
       49BBB826460F677A0000 mov      r11, 0x7A670F4626B8
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
       E93EF5FFFF           jmp      G_M000_IG207
 
G_M000_IG352:                ;; offset=0x2332
       E8794F9AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG353:                ;; offset=0x2338
       4883EC38             sub      rsp, 56
 
G_M000_IG354:                ;; offset=0x233C
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG355:                ;; offset=0x2345
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG356:                ;; offset=0x234D
       4883EC38             sub      rsp, 56
 
G_M000_IG357:                ;; offset=0x2351
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB5C0FDFFFF       mov      rsi, gword ptr [rbp-0x240]
       33D2                 xor      edx, edx
       FF1593C5D5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       90                   nop      
 
G_M000_IG358:                ;; offset=0x236E
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG359:                ;; offset=0x2376
       4883EC38             sub      rsp, 56
 
G_M000_IG360:                ;; offset=0x237A
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG361:                ;; offset=0x2383
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG362:                ;; offset=0x238B
       4883EC38             sub      rsp, 56
 
G_M000_IG363:                ;; offset=0x238F
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB5C8FDFFFF       mov      rsi, gword ptr [rbp-0x238]
       33D2                 xor      edx, edx
       FF1555C5D5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       90                   nop      
 
G_M000_IG364:                ;; offset=0x23AC
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG365:                ;; offset=0x23B4
       4883EC38             sub      rsp, 56
 
G_M000_IG366:                ;; offset=0x23B8
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG367:                ;; offset=0x23C1
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG368:                ;; offset=0x23C9
       4883EC38             sub      rsp, 56
 
G_M000_IG369:                ;; offset=0x23CD
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       4C8B37               mov      r14, gword ptr [rdi]
       4883BDD0FDFFFF00     cmp      gword ptr [rbp-0x230], 0
       750C                 jne      SHORT G_M000_IG371
 
G_M000_IG370:                ;; offset=0x23E4
       BF02000000           mov      edi, 2
       FF157926AEFF         call     [System.ThrowHelper:ThrowArgumentNullException(int)]
       CC                   int3     
 
G_M000_IG371:                ;; offset=0x23F0
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
       FFCF                 dec      edi
       83CF0F               or       edi, 15
       33DB                 xor      ebx, ebx
       F30FBDDF             lzcnt    ebx, edi
       83F31F               xor      ebx, 31
       83C3FD               add      ebx, -3
       48BFE8ECA98E677A0000 mov      rdi, 0x7A678EA9ECE8
       48B820B83C8F677A0000 mov      rax, 0x7A678F3CB820
       FFD0                 call     rax
       833809               cmp      dword ptr [rax], 9
       7E0D                 jle      SHORT G_M000_IG372
       488B7808             mov      rdi, gword ptr [rax+0x08]
       488B4748             mov      rax, bword ptr [rdi+0x48]
       4885C0               test     rax, rax
       750A                 jne      SHORT G_M000_IG373
 
G_M000_IG372:                ;; offset=0x2433
       BF09000000           mov      edi, 9
       E833CEF0FF           call     CORINFO_HELP_GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED
 
G_M000_IG373:                ;; offset=0x243D
       488B4010             mov      rax, gword ptr [rax+0x10]
       4885C0               test     rax, rax
       7509                 jne      SHORT G_M000_IG375
 
G_M000_IG374:                ;; offset=0x2446
       498BFE               mov      rdi, r14
       FF157125CCFF         call     [System.Buffers.SharedArrayPool`1[float]:InitializeTlsBucketsAndTrimming():System.Buffers.SharedArrayPoolThreadLocalArray[]:this]
 
G_M000_IG375:                ;; offset=0x244F
       4533FF               xor      r15d, r15d
       41BD01000000         mov      r13d, 1
       395808               cmp      dword ptr [rax+0x08], ebx
       0F863C020000         jbe      G_M000_IG393
 
G_M000_IG376:                ;; offset=0x2461
       41BF01000000         mov      r15d, 1
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       BE10000000           mov      esi, 16
       C4E261F7F6           shlx     esi, esi, ebx
       397708               cmp      dword ptr [rdi+0x08], esi
       7448                 je       SHORT G_M000_IG378
 
G_M000_IG377:                ;; offset=0x247D
       48BF28CD4810677A0000 mov      rdi, 0x7A671048CD28
       E8F4DFF57C           call     CORINFO_HELP_NEWSFAST
       4C8BE0               mov      r12, rax
       FF154325CCFF         call     [System.SR:get_ArgumentException_BufferNotFromPool():System.String]
       488BD8               mov      rbx, rax
       BF6D040000           mov      edi, 0x46D
       48BE0040450F677A0000 mov      rsi, 0x7A670F454000
       FF159B7A61FF         call     [CORINFO_HELP_STRCNS]
       488BD0               mov      rdx, rax
       488BF3               mov      rsi, rbx
       498BFC               mov      rdi, r12
       FF15AC7B61FF         call     [System.ArgumentException:.ctor(System.String,System.String):this]
       498BFC               mov      rdi, r12
       E85C0CE17C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG378:                ;; offset=0x24C5
       3B5808               cmp      ebx, dword ptr [rax+0x08]
       0F83A2020000         jae      G_M000_IG397
       8BFB                 mov      edi, ebx
       48C1E704             shl      rdi, 4
       4C8D643810           lea      r12, bword ptr [rax+rdi+0x10]
       498B0424             mov      rax, gword ptr [r12]
       488985B8FDFFFF       mov      gword ptr [rbp-0x248], rax
       488BB5D0FDFFFF       mov      rsi, gword ptr [rbp-0x230]
       498BFC               mov      rdi, r12
       E88D369AFD           call     CORINFO_HELP_ASSIGN_REF
       33FF                 xor      edi, edi
       41897C2408           mov      dword ptr [r12+0x08], edi
       4C8BA5B8FDFFFF       mov      r12, gword ptr [rbp-0x248]
       4D85E4               test     r12, r12
       0F8493010000         je       G_M000_IG393
 
G_M000_IG379:                ;; offset=0x250A
       498B7E10             mov      rdi, gword ptr [r14+0x10]
       3B5F08               cmp      ebx, dword ptr [rdi+0x08]
       0F8359020000         jae      G_M000_IG397
       8BF3                 mov      esi, ebx
       488B44F710           mov      rax, gword ptr [rdi+8*rsi+0x10]
       4885C0               test     rax, rax
       750B                 jne      SHORT G_M000_IG380
       498BFE               mov      rdi, r14
       8BF3                 mov      esi, ebx
       FF15DA24CCFF         call     [System.Buffers.SharedArrayPool`1[float]:CreatePerCorePartitions(int):System.Buffers.SharedArrayPoolPartitions:this]
 
G_M000_IG380:                ;; offset=0x252E
       4C8B6808             mov      r13, gword ptr [rax+0x08]
       48BF10AE9E11677A0000 mov      rdi, 0x7A67119EAE10
       FF15268D9AFE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       803D53E99BFD00       cmp      byte  ptr [(reloc 0x7a670f45b35c)], 0
       7412                 je       SHORT G_M000_IG381
       C5F877               vzeroupper 
       E84DBF99FE           call     Interop+Sys:SchedGetCpu():int
       8BD0                 mov      edx, eax
       899570FEFFFF         mov      dword ptr [rbp-0x190], edx
       EB4B                 jmp      SHORT G_M000_IG383
 
G_M000_IG381:                ;; offset=0x255D
       BF0A000000           mov      edi, 10
       FF15B01DE1FF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B4010               mov      eax, dword ptr [rax+0x10]
       89856CFEFFFF         mov      dword ptr [rbp-0x194], eax
       BF0A000000           mov      edi, 10
       FF159C1DE1FF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B956CFEFFFF         mov      edx, dword ptr [rbp-0x194]
       8D4AFF               lea      ecx, [rdx-0x01]
       894810               mov      dword ptr [rax+0x10], ecx
       0FB7C2               movzx    rax, dx
       85C0                 test     eax, eax
       7510                 jne      SHORT G_M000_IG382
       FF159B1DE1FF         call     [System.Threading.ProcessorIdCache:RefreshCurrentProcessorId():int]
       8BD0                 mov      edx, eax
       899570FEFFFF         mov      dword ptr [rbp-0x190], edx
       EB09                 jmp      SHORT G_M000_IG383
 
G_M000_IG382:                ;; offset=0x259F
       C1FA10               sar      edx, 16
       899570FEFFFF         mov      dword ptr [rbp-0x190], edx
 
G_M000_IG383:                ;; offset=0x25A8
       48BFB0AC9E11677A0000 mov      rdi, 0x7A67119EACB0
       FF15B08C9AFE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       8B8570FEFFFF         mov      eax, dword ptr [rbp-0x190]
       33D2                 xor      edx, edx
       F735CAE89BFD         div      edx:eax, dword ptr [(reloc 0x7a670f45b350)]
       8BC2                 mov      eax, edx
       33C9                 xor      ecx, ecx
       E9B1000000           jmp      G_M000_IG390
 
G_M000_IG384:                ;; offset=0x25CF
       413B4508             cmp      eax, dword ptr [r13+0x08]
       0F8397010000         jae      G_M000_IG397
       898578FEFFFF         mov      dword ptr [rbp-0x188], eax
       8BF8                 mov      edi, eax
       498B54FD10           mov      rdx, gword ptr [r13+8*rdi+0x10]
       488995B0FDFFFF       mov      gword ptr [rbp-0x250], rdx
       3812                 cmp      byte  ptr [rdx], dl
       33F6                 xor      esi, esi
       89B568FEFFFF         mov      dword ptr [rbp-0x198], esi
       488BFA               mov      rdi, rdx
       FF15A0A2F6FF         call     [System.Threading.Monitor:Enter(System.Object)]
       488B85B0FDFFFF       mov      rax, gword ptr [rbp-0x250]
       488B7808             mov      rdi, gword ptr [rax+0x08]
       8B4810               mov      ecx, dword ptr [rax+0x10]
       898D64FEFFFF         mov      dword ptr [rbp-0x19C], ecx
       394F08               cmp      dword ptr [rdi+0x08], ecx
       7635                 jbe      SHORT G_M000_IG386
       85C9                 test     ecx, ecx
       7545                 jne      SHORT G_M000_IG387
       33F6                 xor      esi, esi
       897014               mov      dword ptr [rax+0x14], esi
 
G_M000_IG385:                ;; offset=0x2622
       4863F1               movsxd   rsi, ecx
       488D7CF710           lea      rdi, bword ptr [rdi+8*rsi+0x10]
       498BF4               mov      rsi, r12
       E84E359AFD           call     CORINFO_HELP_ASSIGN_REF
       8BBD64FEFFFF         mov      edi, dword ptr [rbp-0x19C]
       FFC7                 inc      edi
       488B85B0FDFFFF       mov      rax, gword ptr [rbp-0x250]
       897810               mov      dword ptr [rax+0x10], edi
       C78568FEFFFF01000000 mov      dword ptr [rbp-0x198], 1
 
G_M000_IG386:                ;; offset=0x264E
       488BF8               mov      rdi, rax
       FF15F19C9AFE         call     [System.Threading.Monitor:Exit(System.Object)]
       83BD68FEFFFF00       cmp      dword ptr [rbp-0x198], 0
       7404                 je       SHORT G_M000_IG388
       EB30                 jmp      SHORT G_M000_IG391
 
G_M000_IG387:                ;; offset=0x2662
       EBBE                 jmp      SHORT G_M000_IG385
 
G_M000_IG388:                ;; offset=0x2664
       8B8578FEFFFF         mov      eax, dword ptr [rbp-0x188]
       FFC0                 inc      eax
       8BF8                 mov      edi, eax
       41397D08             cmp      dword ptr [r13+0x08], edi
       7502                 jne      SHORT G_M000_IG389
       33FF                 xor      edi, edi
 
G_M000_IG389:                ;; offset=0x2676
       8B8D74FEFFFF         mov      ecx, dword ptr [rbp-0x18C]
       FFC1                 inc      ecx
       8BC7                 mov      eax, edi
 
G_M000_IG390:                ;; offset=0x2680
       898D74FEFFFF         mov      dword ptr [rbp-0x18C], ecx
       41394D08             cmp      dword ptr [r13+0x08], ecx
       0F8F3FFFFFFF         jg       G_M000_IG384
       EB08                 jmp      SHORT G_M000_IG392
 
G_M000_IG391:                ;; offset=0x2692
       41BD01000000         mov      r13d, 1
       EB03                 jmp      SHORT G_M000_IG393
 
G_M000_IG392:                ;; offset=0x269A
       4533ED               xor      r13d, r13d
 
G_M000_IG393:                ;; offset=0x269D
       48BFF80180295F7A0000 mov      rdi, 0x7A5F298001F8
       4C8B27               mov      r12, gword ptr [rdi]
       4180BC249D00000000   cmp      byte  ptr [r12+0x9D], 0
       0F84BD000000         je       G_M000_IG398
 
G_M000_IG394:                ;; offset=0x26B9
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       0F84AC000000         je       G_M000_IG398
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       FF1539A173FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       898584FEFFFF         mov      dword ptr [rbp-0x17C], eax
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       8B4F08               mov      ecx, dword ptr [rdi+0x08]
       898D80FEFFFF         mov      dword ptr [rbp-0x180], ecx
       498BFE               mov      rdi, r14
       FF151AA173FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       448BC0               mov      r8d, eax
       498BFC               mov      rdi, r12
       8B9584FEFFFF         mov      edx, dword ptr [rbp-0x17C]
       8B8D80FEFFFF         mov      ecx, dword ptr [rbp-0x180]
       BE03000000           mov      esi, 3
       FF15E532E1FF         call     [System.Diagnostics.Tracing.EventSource:WriteEvent(int,int,int,int):this]
       4585FD               test     r15d, r13d
       755E                 jne      SHORT G_M000_IG398
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       FF15EBA073FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       448BE8               mov      r13d, eax
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       8B4708               mov      eax, dword ptr [rdi+0x08]
       89857CFEFFFF         mov      dword ptr [rbp-0x184], eax
       498BFE               mov      rdi, r14
       FF15CFA073FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       4585FF               test     r15d, r15d
       750E                 jne      SHORT G_M000_IG395
       41B8FFFFFFFF         mov      r8d, -1
       41B901000000         mov      r9d, 1
       EB06                 jmp      SHORT G_M000_IG396
 
G_M000_IG395:                ;; offset=0x2756
       448BC3               mov      r8d, ebx
       4533C9               xor      r9d, r9d
 
G_M000_IG396:                ;; offset=0x275C
       498BFC               mov      rdi, r12
       418BF5               mov      esi, r13d
       8B957CFEFFFF         mov      edx, dword ptr [rbp-0x184]
       FF15E222CCFF         call     [System.Buffers.ArrayPoolEventSource:BufferDropped(int,int,int,int,int):this]
       EB06                 jmp      SHORT G_M000_IG398
 
G_M000_IG397:                ;; offset=0x2770
       E83B4B9AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG398:                ;; offset=0x2776
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG399:                ;; offset=0x277E
       4883EC38             sub      rsp, 56
 
G_M000_IG400:                ;; offset=0x2782
       48837D9800           cmp      qword ptr [rbp-0x68], 0
       7431                 je       SHORT G_M000_IG405
 
G_M000_IG401:                ;; offset=0x2789
       488D7D98             lea      rdi, bword ptr [rbp-0x68]
       33C0                 xor      eax, eax
       488BD8               mov      rbx, rax
       48871F               xchg     qword ptr [rdi], rbx
       4885DB               test     rbx, rbx
       7507                 jne      SHORT G_M000_IG403
 
G_M000_IG402:                ;; offset=0x279A
       FF15180CE1FF         call     [System.ThrowHelper:ThrowInvalidOperationException_HandleIsNotInitialized()]
       CC                   int3     
 
G_M000_IG403:                ;; offset=0x27A1
       4883E3FE             and      rbx, -2
       488BFB               mov      rdi, rbx
       E8234FE17C           call     System.Runtime.InteropServices.GCHandle:_InternalFree(nint):bool
       85C0                 test     eax, eax
       7509                 jne      SHORT G_M000_IG405
 
G_M000_IG404:                ;; offset=0x27B1
       488BFB               mov      rdi, rbx
       FF15160CE1FF         call     [System.Runtime.InteropServices.GCHandle:InternalFreeWithGCTransition(nint)]
 
G_M000_IG405:                ;; offset=0x27BA
       48837D8800           cmp      gword ptr [rbp-0x78], 0
       7417                 je       SHORT G_M000_IG407
 
G_M000_IG406:                ;; offset=0x27C1
       488B7D88             mov      rdi, gword ptr [rbp-0x78]
       49BBA826460F677A0000 mov      r11, 0x7A670F4626A8
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33FF                 xor      rdi, rdi
       48897D88             mov      gword ptr [rbp-0x78], rdi
 
G_M000_IG407:                ;; offset=0x27D8
       33FF                 xor      edi, edi
       48897D90             mov      qword ptr [rbp-0x70], rdi
 
G_M000_IG408:                ;; offset=0x27DE
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG409:                ;; offset=0x27E6
       4883EC38             sub      rsp, 56
 
G_M000_IG410:                ;; offset=0x27EA
       48837DB000           cmp      qword ptr [rbp-0x50], 0
       7431                 je       SHORT G_M000_IG415
 
G_M000_IG411:                ;; offset=0x27F1
       488D7DB0             lea      rdi, bword ptr [rbp-0x50]
       33C0                 xor      eax, eax
       488BD8               mov      rbx, rax
       48871F               xchg     qword ptr [rdi], rbx
       4885DB               test     rbx, rbx
       7507                 jne      SHORT G_M000_IG413
 
G_M000_IG412:                ;; offset=0x2802
       FF15B00BE1FF         call     [System.ThrowHelper:ThrowInvalidOperationException_HandleIsNotInitialized()]
       CC                   int3     
 
G_M000_IG413:                ;; offset=0x2809
       4883E3FE             and      rbx, -2
       488BFB               mov      rdi, rbx
       E8BB4EE17C           call     System.Runtime.InteropServices.GCHandle:_InternalFree(nint):bool
       85C0                 test     eax, eax
       7509                 jne      SHORT G_M000_IG415
 
G_M000_IG414:                ;; offset=0x2819
       488BFB               mov      rdi, rbx
       FF15AE0BE1FF         call     [System.Runtime.InteropServices.GCHandle:InternalFreeWithGCTransition(nint)]
 
G_M000_IG415:                ;; offset=0x2822
       48837DA000           cmp      gword ptr [rbp-0x60], 0
       7417                 je       SHORT G_M000_IG417
 
G_M000_IG416:                ;; offset=0x2829
       488B7DA0             mov      rdi, gword ptr [rbp-0x60]
       49BBB026460F677A0000 mov      r11, 0x7A670F4626B0
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG417:                ;; offset=0x2840
       33FF                 xor      edi, edi
       48897DA8             mov      qword ptr [rbp-0x58], rdi
 
G_M000_IG418:                ;; offset=0x2846
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG419:                ;; offset=0x284E
       4883EC38             sub      rsp, 56
 
G_M000_IG420:                ;; offset=0x2852
       48837DC800           cmp      qword ptr [rbp-0x38], 0
       7431                 je       SHORT G_M000_IG425
 
G_M000_IG421:                ;; offset=0x2859
       488D7DC8             lea      rdi, bword ptr [rbp-0x38]
       33C0                 xor      eax, eax
       488BD8               mov      rbx, rax
       48871F               xchg     qword ptr [rdi], rbx
       4885DB               test     rbx, rbx
       7507                 jne      SHORT G_M000_IG423
 
G_M000_IG422:                ;; offset=0x286A
       FF15480BE1FF         call     [System.ThrowHelper:ThrowInvalidOperationException_HandleIsNotInitialized()]
       CC                   int3     
 
G_M000_IG423:                ;; offset=0x2871
       4883E3FE             and      rbx, -2
       488BFB               mov      rdi, rbx
       E8534EE17C           call     System.Runtime.InteropServices.GCHandle:_InternalFree(nint):bool
       85C0                 test     eax, eax
       7509                 jne      SHORT G_M000_IG425
 
G_M000_IG424:                ;; offset=0x2881
       488BFB               mov      rdi, rbx
       FF15460BE1FF         call     [System.Runtime.InteropServices.GCHandle:InternalFreeWithGCTransition(nint)]
 
G_M000_IG425:                ;; offset=0x288A
       48837DB800           cmp      gword ptr [rbp-0x48], 0
       7417                 je       SHORT G_M000_IG427
 
G_M000_IG426:                ;; offset=0x2891
       488B7DB8             mov      rdi, gword ptr [rbp-0x48]
       49BBB826460F677A0000 mov      r11, 0x7A670F4626B8
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33FF                 xor      rdi, rdi
       48897DB8             mov      gword ptr [rbp-0x48], rdi
 
G_M000_IG427:                ;; offset=0x28A8
       33FF                 xor      edi, edi
       48897DC0             mov      qword ptr [rbp-0x40], rdi
 
G_M000_IG428:                ;; offset=0x28AE
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
; Total bytes of code 10422
