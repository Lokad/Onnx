; Assembly listing for method Lokad.Onnx.Tensor`1[float]:RunBatchedFloatMatMul(Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],Lokad.Onnx.TensorExecutionOptions) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 122268
; 63 inlinees with PGO data; 144 single block inlinees; 3 inlinees without PGO data

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
       48BF50065F6B947C0000 mov      rdi, 0x7C946B5F0650
       E8B6FCF57C           call     CORINFO_HELP_NEWSFAST
       48898528FEFFFF       mov      gword ptr [rbp-0x1D8], rax
       488D7848             lea      rdi, bword ptr [rax+0x48]
       40383F               cmp      byte  ptr [rdi], dil
       488D7510             lea      rsi, [rbp+0x10]
       BA30000000           mov      edx, 48
       C5F877               vzeroupper 
       FF15DEAA9AFE         call     [CORINFO_HELP_BULK_WRITEBARRIER]
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       4C8B6850             mov      r13, gword ptr [rax+0x50]
       498BFF               mov      rdi, r15
       4885FF               test     rdi, rdi
       7415                 je       SHORT G_M000_IG04
 
G_M000_IG03:                ;; offset=0x009D
       48BE20403E6B947C0000 mov      rsi, 0x7C946B3E4020
       483937               cmp      qword ptr [rdi], rsi
       0F85891A0000         jne      G_M000_IG258
       33FF                 xor      rdi, rdi
 
G_M000_IG04:                ;; offset=0x00B2
       4885FF               test     rdi, rdi
       0F85981A0000         jne      G_M000_IG259
 
G_M000_IG05:                ;; offset=0x00BB
       4D8BE7               mov      r12, r15
       4D85E4               test     r12, r12
       7414                 je       SHORT G_M000_IG07
 
G_M000_IG06:                ;; offset=0x00C3
       48BF20403E6B947C0000 mov      rdi, 0x7C946B3E4020
       49393C24             cmp      qword ptr [r12], rdi
       0F858F1A0000         jne      G_M000_IG260
 
G_M000_IG07:                ;; offset=0x00D7
       4D85E4               test     r12, r12
       0F84F71B0000         je       G_M000_IG276
       41807C243C00         cmp      byte  ptr [r12+0x3C], 0
       0F85EB1B0000         jne      G_M000_IG276
       498B7C2418           mov      rdi, gword ptr [r12+0x18]
       4885FF               test     rdi, rdi
       0F847C1A0000         je       G_M000_IG261
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       448B4708             mov      r8d, dword ptr [rdi+0x08]
 
G_M000_IG08:                ;; offset=0x0102
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       4885FF               test     rdi, rdi
       0F848C1A0000         je       G_M000_IG262
       488D7710             lea      rsi, bword ptr [rdi+0x10]
       8B5708               mov      edx, dword ptr [rdi+0x08]
       48898DA8FDFFFF       mov      bword ptr [rbp-0x258], rcx
       44898560FEFFFF       mov      dword ptr [rbp-0x1A0], r8d
 
G_M000_IG09:                ;; offset=0x0125
       488BFE               mov      rdi, rsi
       8BF2                 mov      esi, edx
       33D2                 xor      edx, edx
       FF15BE9761FF         call     [Lokad.Onnx.ArrayUtilities:GetStrides(System.ReadOnlySpan`1[int],bool):int[]]
       4885C0               test     rax, rax
       0F84781A0000         je       G_M000_IG263
       488D7810             lea      rdi, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG10:                ;; offset=0x0142
       398560FEFFFF         cmp      dword ptr [rbp-0x1A0], eax
       0F85221B0000         jne      G_M000_IG274
       488B8DA8FDFFFF       mov      rcx, bword ptr [rbp-0x258]
       488BD1               mov      rdx, rcx
       488BF7               mov      rsi, rdi
       8BC0                 mov      eax, eax
       48C1E002             shl      rax, 2
       4883F808             cmp      rax, 8
       0F82511A0000         jb       G_M000_IG264
 
G_M000_IG11:                ;; offset=0x016B
       483BD6               cmp      rdx, rsi
       7456                 je       SHORT G_M000_IG17
 
G_M000_IG12:                ;; offset=0x0170
       4883F840             cmp      rax, 64
       0F828C160000         jb       G_M000_IG204
 
G_M000_IG13:                ;; offset=0x017A
       33C9                 xor      ecx, ecx
       4883C0C0             add      rax, -64
       7428                 je       SHORT G_M000_IG16
 
G_M000_IG14:                ;; offset=0x0182
       62F17C481002         vmovups  zmm0, zmmword ptr [rdx]
       62F37D483E0E04       vpcmpfalseub k1, zmm0, zmmword ptr [rsi]
       C4E1F898C9           kortestq k1, k1
       0F851E170000         jne      G_M000_IG216
 
G_M000_IG15:                ;; offset=0x019D
       4883C140             add      rcx, 64
       483BC1               cmp      rax, rcx
       0F87661A0000         ja       G_M000_IG271
 
G_M000_IG16:                ;; offset=0x01AA
       62F17C48100402       vmovups  zmm0, zmmword ptr [rdx+rax]
       62F17D48740C06       vpcmpeqb k1, zmm0, zmmword ptr [rsi+rax]
       C4E1F898C9           kortestq k1, k1
       0F83F5160000         jae      G_M000_IG216
 
G_M000_IG17:                ;; offset=0x01C6
       BF01000000           mov      edi, 1
 
G_M000_IG18:                ;; offset=0x01CB
       85FF                 test     edi, edi
       0F84041B0000         je       G_M000_IG276
       418B7C245C           mov      edi, dword ptr [r12+0x5C]
       413B7C2430           cmp      edi, dword ptr [r12+0x30]
       0F85941A0000         jne      G_M000_IG275
 
G_M000_IG19:                ;; offset=0x01E3
       4D8BFC               mov      r15, r12
 
G_M000_IG20:                ;; offset=0x01E6
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       4C8B6850             mov      r13, gword ptr [rax+0x50]
       488BFB               mov      rdi, rbx
       4885FF               test     rdi, rdi
       7415                 je       SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x01F9
       48BE20403E6B947C0000 mov      rsi, 0x7C946B3E4020
       483937               cmp      qword ptr [rdi], rsi
       0F85EC1A0000         jne      G_M000_IG277
       33FF                 xor      rdi, rdi
 
G_M000_IG22:                ;; offset=0x020E
       4885FF               test     rdi, rdi
       0F85FB1A0000         jne      G_M000_IG278
 
G_M000_IG23:                ;; offset=0x0217
       4C8BE3               mov      r12, rbx
       4D85E4               test     r12, r12
       7414                 je       SHORT G_M000_IG25
 
G_M000_IG24:                ;; offset=0x021F
       48BF20403E6B947C0000 mov      rdi, 0x7C946B3E4020
       49393C24             cmp      qword ptr [r12], rdi
       0F85F21A0000         jne      G_M000_IG279
 
G_M000_IG25:                ;; offset=0x0233
       4D85E4               test     r12, r12
       0F845A1C0000         je       G_M000_IG295
       41807C243C00         cmp      byte  ptr [r12+0x3C], 0
       0F854E1C0000         jne      G_M000_IG295
       498B7C2418           mov      rdi, gword ptr [r12+0x18]
       4885FF               test     rdi, rdi
       0F84DF1A0000         je       G_M000_IG280
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       448B4708             mov      r8d, dword ptr [rdi+0x08]
 
G_M000_IG26:                ;; offset=0x025E
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       4885FF               test     rdi, rdi
       0F84EF1A0000         je       G_M000_IG281
       488D7710             lea      rsi, bword ptr [rdi+0x10]
       8B5708               mov      edx, dword ptr [rdi+0x08]
       48898DA0FDFFFF       mov      bword ptr [rbp-0x260], rcx
       4489855CFEFFFF       mov      dword ptr [rbp-0x1A4], r8d
 
G_M000_IG27:                ;; offset=0x0281
       488BFE               mov      rdi, rsi
       8BF2                 mov      esi, edx
       33D2                 xor      edx, edx
       FF15629661FF         call     [Lokad.Onnx.ArrayUtilities:GetStrides(System.ReadOnlySpan`1[int],bool):int[]]
       4885C0               test     rax, rax
       0F84DB1A0000         je       G_M000_IG282
       488D7810             lea      rdi, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG28:                ;; offset=0x029E
       39855CFEFFFF         cmp      dword ptr [rbp-0x1A4], eax
       0F85851B0000         jne      G_M000_IG293
       488B8DA0FDFFFF       mov      rcx, bword ptr [rbp-0x260]
       488BD1               mov      rdx, rcx
       488BF7               mov      rsi, rdi
       8BC0                 mov      eax, eax
       48C1E002             shl      rax, 2
       4883F808             cmp      rax, 8
       0F82B41A0000         jb       G_M000_IG283
 
G_M000_IG29:                ;; offset=0x02C7
       483BD6               cmp      rdx, rsi
       7456                 je       SHORT G_M000_IG35
 
G_M000_IG30:                ;; offset=0x02CC
       4883F840             cmp      rax, 64
       0F82EC150000         jb       G_M000_IG217
 
G_M000_IG31:                ;; offset=0x02D6
       33C9                 xor      ecx, ecx
       4883C0C0             add      rax, -64
       7428                 je       SHORT G_M000_IG34
 
G_M000_IG32:                ;; offset=0x02DE
       62F17C481002         vmovups  zmm0, zmmword ptr [rdx]
       62F37D483E0E04       vpcmpfalseub k1, zmm0, zmmword ptr [rsi]
       C4E1F898C9           kortestq k1, k1
       0F857E160000         jne      G_M000_IG229
 
G_M000_IG33:                ;; offset=0x02F9
       4883C140             add      rcx, 64
       483BC1               cmp      rax, rcx
       0F87C91A0000         ja       G_M000_IG290
 
G_M000_IG34:                ;; offset=0x0306
       62F17C48100402       vmovups  zmm0, zmmword ptr [rdx+rax]
       62F17D48740C06       vpcmpeqb k1, zmm0, zmmword ptr [rsi+rax]
       C4E1F898C9           kortestq k1, k1
       0F8355160000         jae      G_M000_IG229
 
G_M000_IG35:                ;; offset=0x0322
       BF01000000           mov      edi, 1
 
G_M000_IG36:                ;; offset=0x0327
       85FF                 test     edi, edi
       0F84671B0000         je       G_M000_IG295
       418B7C245C           mov      edi, dword ptr [r12+0x5C]
       413B7C2430           cmp      edi, dword ptr [r12+0x30]
       0F85F71A0000         jne      G_M000_IG294
 
G_M000_IG37:                ;; offset=0x033F
       498BDC               mov      rbx, r12
 
G_M000_IG38:                ;; offset=0x0342
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       4C8B6850             mov      r13, gword ptr [rax+0x50]
       4D8BE6               mov      r12, r14
       4D85E4               test     r12, r12
       7414                 je       SHORT G_M000_IG40
 
G_M000_IG39:                ;; offset=0x0355
       48BF20403E6B947C0000 mov      rdi, 0x7C946B3E4020
       49393C24             cmp      qword ptr [r12], rdi
       0F854E1B0000         jne      G_M000_IG296
 
G_M000_IG40:                ;; offset=0x0369
       4D85E4               test     r12, r12
       0F84B61C0000         je       G_M000_IG312
       41807C243C00         cmp      byte  ptr [r12+0x3C], 0
       0F85AA1C0000         jne      G_M000_IG312
       498B7C2418           mov      rdi, gword ptr [r12+0x18]
       4885FF               test     rdi, rdi
       0F843B1B0000         je       G_M000_IG297
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       448B4708             mov      r8d, dword ptr [rdi+0x08]
 
G_M000_IG41:                ;; offset=0x0394
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       4885FF               test     rdi, rdi
       0F844B1B0000         je       G_M000_IG298
       488D7710             lea      rsi, bword ptr [rdi+0x10]
       8B5708               mov      edx, dword ptr [rdi+0x08]
       48898D98FDFFFF       mov      bword ptr [rbp-0x268], rcx
       44898558FEFFFF       mov      dword ptr [rbp-0x1A8], r8d
 
G_M000_IG42:                ;; offset=0x03B7
       488BFE               mov      rdi, rsi
       8BF2                 mov      esi, edx
       33D2                 xor      edx, edx
       FF152C9561FF         call     [Lokad.Onnx.ArrayUtilities:GetStrides(System.ReadOnlySpan`1[int],bool):int[]]
       4885C0               test     rax, rax
       0F84371B0000         je       G_M000_IG299
       488D7810             lea      rdi, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG43:                ;; offset=0x03D4
       398558FEFFFF         cmp      dword ptr [rbp-0x1A8], eax
       0F85E11B0000         jne      G_M000_IG310
       488B8D98FDFFFF       mov      rcx, bword ptr [rbp-0x268]
       488BD1               mov      rdx, rcx
       488BF7               mov      rsi, rdi
       8BC0                 mov      eax, eax
       48C1E002             shl      rax, 2
       4883F808             cmp      rax, 8
       0F82101B0000         jb       G_M000_IG300
 
G_M000_IG44:                ;; offset=0x03FD
       483BD6               cmp      rdx, rsi
       7456                 je       SHORT G_M000_IG50
 
G_M000_IG45:                ;; offset=0x0402
       4883F840             cmp      rax, 64
       0F8272150000         jb       G_M000_IG230
 
G_M000_IG46:                ;; offset=0x040C
       33C9                 xor      ecx, ecx
       4883C0C0             add      rax, -64
       7428                 je       SHORT G_M000_IG49
 
G_M000_IG47:                ;; offset=0x0414
       62F17C481002         vmovups  zmm0, zmmword ptr [rdx]
       62F37D483E0E04       vpcmpfalseub k1, zmm0, zmmword ptr [rsi]
       C4E1F898C9           kortestq k1, k1
       0F8504160000         jne      G_M000_IG242
 
G_M000_IG48:                ;; offset=0x042F
       4883C140             add      rcx, 64
       483BC1               cmp      rax, rcx
       0F87251B0000         ja       G_M000_IG307
 
G_M000_IG49:                ;; offset=0x043C
       62F17C48100402       vmovups  zmm0, zmmword ptr [rdx+rax]
       62F17D48740C06       vpcmpeqb k1, zmm0, zmmword ptr [rsi+rax]
       C4E1F898C9           kortestq k1, k1
       0F83DB150000         jae      G_M000_IG242
 
G_M000_IG50:                ;; offset=0x0458
       BF01000000           mov      edi, 1
 
G_M000_IG51:                ;; offset=0x045D
       85FF                 test     edi, edi
       0F84C31B0000         je       G_M000_IG312
       418B7C245C           mov      edi, dword ptr [r12+0x5C]
       413B7C2430           cmp      edi, dword ptr [r12+0x30]
       0F85531B0000         jne      G_M000_IG311
 
G_M000_IG52:                ;; offset=0x0475
       4D8BF4               mov      r14, r12
       498B7710             mov      rsi, gword ptr [r15+0x10]
       4885F6               test     rsi, rsi
       0F84C41B0000         je       G_M000_IG313
       488D7E10             lea      rdi, bword ptr [rsi+0x10]
       8B7608               mov      esi, dword ptr [rsi+0x08]
 
G_M000_IG53:                ;; offset=0x048C
       4889BD78FFFFFF       mov      bword ptr [rbp-0x88], rdi
       897580               mov      dword ptr [rbp-0x80], esi
       448B6D80             mov      r13d, dword ptr [rbp-0x80]
       4183C5FE             add      r13d, -2
       443B6D80             cmp      r13d, dword ptr [rbp-0x80]
       0F87AA1B0000         ja       G_M000_IG314
       4C8BA578FFFFFF       mov      r12, bword ptr [rbp-0x88]
       4585ED               test     r13d, r13d
       0F84A11B0000         je       G_M000_IG315
       4963F5               movsxd   rsi, r13d
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E866F9F57C           call     CORINFO_HELP_NEWARR_1_VC
       488BF8               mov      rdi, rax
       4889BDF0FDFFFF       mov      gword ptr [rbp-0x210], rdi
       418BD5               mov      edx, r13d
       48C1E202             shl      rdx, 2
       4883C710             add      rdi, 16
       498BF4               mov      rsi, r12
       FF15F8A69AFE         call     [System.SpanHelpers:Memmove(byref,byref,nuint)]
       4C8BADF0FDFFFF       mov      r13, gword ptr [rbp-0x210]
 
G_M000_IG54:                ;; offset=0x04EF
       4C89AD20FEFFFF       mov      gword ptr [rbp-0x1E0], r13
       498B7710             mov      rsi, gword ptr [r15+0x10]
       488BFE               mov      rdi, rsi
       4885FF               test     rdi, rdi
       0F84621B0000         je       G_M000_IG316
       488D4710             lea      rax, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG55:                ;; offset=0x050D
       8D4FFE               lea      ecx, [rdi-0x02]
       3BCF                 cmp      ecx, edi
       0F83C91D0000         jae      G_M000_IG346
       8BF9                 mov      edi, ecx
       448B2CB8             mov      r13d, dword ptr [rax+4*rdi]
       488B8528FEFFFF       mov      rax, gword ptr [rbp-0x1D8]
       44896838             mov      dword ptr [rax+0x38], r13d
       488BFE               mov      rdi, rsi
       4885FF               test     rdi, rdi
       0F843C1B0000         je       G_M000_IG317
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG56:                ;; offset=0x053C
       8D57FF               lea      edx, [rdi-0x01]
       3BD7                 cmp      edx, edi
       0F839A1D0000         jae      G_M000_IG346
       8BFA                 mov      edi, edx
       8B3CB9               mov      edi, dword ptr [rcx+4*rdi]
       89783C               mov      dword ptr [rax+0x3C], edi
       488B7B10             mov      rdi, gword ptr [rbx+0x10]
       4885FF               test     rdi, rdi
       0F841E1B0000         je       G_M000_IG318
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG57:                ;; offset=0x0563
       8D57FF               lea      edx, [rdi-0x01]
       3BD7                 cmp      edx, edi
       0F83731D0000         jae      G_M000_IG346
       8BFA                 mov      edi, edx
       8B3CB9               mov      edi, dword ptr [rcx+4*rdi]
       897840               mov      dword ptr [rax+0x40], edi
       488BBD20FEFFFF       mov      rdi, gword ptr [rbp-0x1E0]
       448B6708             mov      r12d, dword ptr [rdi+0x08]
       4489A554FEFFFF       mov      dword ptr [rbp-0x1AC], r12d
       4885F6               test     rsi, rsi
       0F84F21A0000         je       G_M000_IG319
       488D5610             lea      rdx, bword ptr [rsi+0x10]
       448B4608             mov      r8d, dword ptr [rsi+0x08]
 
G_M000_IG58:                ;; offset=0x0599
       4D8BCF               mov      r9, r15
       48BE20403E6B947C0000 mov      rsi, 0x7C946B3E4020
       493931               cmp      qword ptr [r9], rsi
       0F85FA1A0000         jne      G_M000_IG320
       4533C9               xor      r9, r9
 
G_M000_IG59:                ;; offset=0x05B2
       4D85C9               test     r9, r9
       0F85241B0000         jne      G_M000_IG321
 
G_M000_IG60:                ;; offset=0x05BB
       4D8B5718             mov      r10, gword ptr [r15+0x18]
       4C8995E8FDFFFF       mov      gword ptr [rbp-0x218], r10
 
G_M000_IG61:                ;; offset=0x05C6
       8B8D54FEFFFF         mov      ecx, dword ptr [rbp-0x1AC]
       898D4CFEFFFF         mov      dword ptr [rbp-0x1B4], ecx
       48899588FDFFFF       mov      bword ptr [rbp-0x278], rdx
       44898548FEFFFF       mov      dword ptr [rbp-0x1B8], r8d
       8BF1                 mov      esi, ecx
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E83FF8F57C           call     CORINFO_HELP_NEWARR_1_VC
       488BC8               mov      rcx, rax
       33F6                 xor      esi, esi
       8BBD4CFEFFFF         mov      edi, dword ptr [rbp-0x1B4]
       85FF                 test     edi, edi
       7E5F                 jle      SHORT G_M000_IG71
 
G_M000_IG62:                ;; offset=0x0600
       4C8B95E8FDFFFF       mov      r10, gword ptr [rbp-0x218]
       4D85D2               test     r10, r10
       0F8436140000         je       G_M000_IG244
 
G_M000_IG63:                ;; offset=0x0610
       448B8548FEFFFF       mov      r8d, dword ptr [rbp-0x1B8]
       413BF8               cmp      edi, r8d
       0F8FE41A0000         jg       G_M000_IG324
 
G_M000_IG64:                ;; offset=0x0620
       41397A08             cmp      dword ptr [r10+0x08], edi
       0F8CD51A0000         jl       G_M000_IG323
 
G_M000_IG65:                ;; offset=0x062A
       397908               cmp      dword ptr [rcx+0x08], edi
       0F8CC71A0000         jl       G_M000_IG322
 
G_M000_IG66:                ;; offset=0x0633
       4183C0FE             add      r8d, -2
 
G_M000_IG67:                ;; offset=0x0637
       413BF0               cmp      esi, r8d
       7D13                 jge      SHORT G_M000_IG69
 
G_M000_IG68:                ;; offset=0x063C
       8BC6                 mov      eax, esi
       488B9588FDFFFF       mov      rdx, bword ptr [rbp-0x278]
       833C8201             cmp      dword ptr [rdx+4*rax], 1
       0F85EB130000         jne      G_M000_IG243
 
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
       0F84A51A0000         je       G_M000_IG326
       488D4E10             lea      rcx, bword ptr [rsi+0x10]
       8B5608               mov      edx, dword ptr [rsi+0x08]
 
G_M000_IG72:                ;; offset=0x0681
       4C8BC3               mov      r8, rbx
       48BE20403E6B947C0000 mov      rsi, 0x7C946B3E4020
       493930               cmp      qword ptr [r8], rsi
       0F85AB1A0000         jne      G_M000_IG327
       4533C0               xor      r8, r8
 
G_M000_IG73:                ;; offset=0x069A
       4D85C0               test     r8, r8
       0F85D31A0000         jne      G_M000_IG328
 
G_M000_IG74:                ;; offset=0x06A3
       4C8B4B18             mov      r9, gword ptr [rbx+0x18]
       4C898DE0FDFFFF       mov      gword ptr [rbp-0x220], r9
 
G_M000_IG75:                ;; offset=0x06AE
       8B8544FEFFFF         mov      eax, dword ptr [rbp-0x1BC]
       89853CFEFFFF         mov      dword ptr [rbp-0x1C4], eax
       48898D78FDFFFF       mov      bword ptr [rbp-0x288], rcx
       899538FEFFFF         mov      dword ptr [rbp-0x1C8], edx
       8BF0                 mov      esi, eax
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E858F7F57C           call     CORINFO_HELP_NEWARR_1_VC
       488BF8               mov      rdi, rax
       33F6                 xor      esi, esi
       8B853CFEFFFF         mov      eax, dword ptr [rbp-0x1C4]
       85C0                 test     eax, eax
       7E5C                 jle      SHORT G_M000_IG85
 
G_M000_IG76:                ;; offset=0x06E7
       4C8B8DE0FDFFFF       mov      r9, gword ptr [rbp-0x220]
       4D85C9               test     r9, r9
       0F84A5130000         je       G_M000_IG249
 
G_M000_IG77:                ;; offset=0x06F7
       8B9538FEFFFF         mov      edx, dword ptr [rbp-0x1C8]
       3BC2                 cmp      eax, edx
       0F8F961A0000         jg       G_M000_IG331
 
G_M000_IG78:                ;; offset=0x0705
       41394108             cmp      dword ptr [r9+0x08], eax
       0F8C871A0000         jl       G_M000_IG330
 
G_M000_IG79:                ;; offset=0x070F
       394708               cmp      dword ptr [rdi+0x08], eax
       0F8C791A0000         jl       G_M000_IG329
 
G_M000_IG80:                ;; offset=0x0718
       83C2FE               add      edx, -2
 
G_M000_IG81:                ;; offset=0x071B
       3BF2                 cmp      esi, edx
       7D14                 jge      SHORT G_M000_IG83
 
G_M000_IG82:                ;; offset=0x071F
       8BCE                 mov      ecx, esi
       4C8B8578FDFFFF       mov      r8, bword ptr [rbp-0x288]
       41833C8801           cmp      dword ptr [r8+4*rcx], 1
       0F855D130000         jne      G_M000_IG248
 
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
       0F845C1A0000         je       G_M000_IG333
       488D4E10             lea      rcx, bword ptr [rsi+0x10]
       8B5608               mov      edx, dword ptr [rsi+0x08]
 
G_M000_IG86:                ;; offset=0x0761
       898534FEFFFF         mov      dword ptr [rbp-0x1CC], eax
       48898D70FDFFFF       mov      bword ptr [rbp-0x290], rcx
       899530FEFFFF         mov      dword ptr [rbp-0x1D0], edx
       4D8B4618             mov      r8, gword ptr [r14+0x18]
       4C8985D8FDFFFF       mov      gword ptr [rbp-0x228], r8
       8BF0                 mov      esi, eax
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E8A0F6F57C           call     CORINFO_HELP_NEWARR_1_VC
       4C8BC0               mov      r8, rax
       33C0                 xor      eax, eax
       8B9534FEFFFF         mov      edx, dword ptr [rbp-0x1CC]
       85D2                 test     edx, edx
       7E5C                 jle      SHORT G_M000_IG96
 
G_M000_IG87:                ;; offset=0x079F
       488BBDD8FDFFFF       mov      rdi, gword ptr [rbp-0x228]
       4885FF               test     rdi, rdi
       0F8440130000         je       G_M000_IG254
 
G_M000_IG88:                ;; offset=0x07AF
       8BB530FEFFFF         mov      esi, dword ptr [rbp-0x1D0]
       3BD6                 cmp      edx, esi
       0F8F0C1A0000         jg       G_M000_IG336
 
G_M000_IG89:                ;; offset=0x07BD
       395708               cmp      dword ptr [rdi+0x08], edx
       0F8CFE190000         jl       G_M000_IG335
 
G_M000_IG90:                ;; offset=0x07C6
       41395008             cmp      dword ptr [r8+0x08], edx
       0F8CEF190000         jl       G_M000_IG334
 
G_M000_IG91:                ;; offset=0x07D0
       83C6FE               add      esi, -2
 
G_M000_IG92:                ;; offset=0x07D3
       3BC6                 cmp      eax, esi
       7D14                 jge      SHORT G_M000_IG94
 
G_M000_IG93:                ;; offset=0x07D7
       8BC8                 mov      ecx, eax
       4C8B8D70FDFFFF       mov      r9, bword ptr [rbp-0x290]
       41833C8901           cmp      dword ptr [r9+4*rcx], 1
       0F85F9120000         jne      G_M000_IG253
 
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
       0F8DA1190000         jge      G_M000_IG338
 
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
       0F858E190000         jne      G_M000_IG339
 
G_M000_IG102:                ;; offset=0x0877
       498BF9               mov      rdi, r9
       488BF3               mov      rsi, rbx
       FF159515CCFF         call     [Lokad.Onnx.GraphPacking:ResolvePacked(System.Collections.Generic.IReadOnlyDictionary`2[float[],Lokad.Onnx.PackedMatMulWeight],Lokad.Onnx.Tensor`1[float]):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE8               mov      r13, rax
       4D85ED               test     r13, r13
       0F84FD000000         je       G_M000_IG112
 
G_M000_IG103:                ;; offset=0x088F
       498B7D10             mov      rdi, gword ptr [r13+0x10]
       488BC7               mov      rax, rdi
       4885C0               test     rax, rax
       0F8488190000         je       G_M000_IG340
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG104:                ;; offset=0x08A2
       83F802               cmp      eax, 2
       0F85E1000000         jne      G_M000_IG112
       488BC7               mov      rax, rdi
       4885C0               test     rax, rax
       0F8477190000         je       G_M000_IG341
       488D4810             lea      rcx, bword ptr [rax+0x10]
       8B4008               mov      eax, dword ptr [rax+0x08]
 
G_M000_IG105:                ;; offset=0x08BE
       85C0                 test     eax, eax
       0F841B1A0000         je       G_M000_IG346
       8B01                 mov      eax, dword ptr [rcx]
       4885FF               test     rdi, rdi
       0F8466190000         je       G_M000_IG342
       488D4F10             lea      rcx, bword ptr [rdi+0x10]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
 
G_M000_IG106:                ;; offset=0x08D8
       83FF01               cmp      edi, 1
       0F86001A0000         jbe      G_M000_IG346
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
       0F851C190000         jne      G_M000_IG343
 
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
       FF15EA29CCFF         call     [Lokad.Onnx.Tensor`1[float]:RunPackedBatches(Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],int[],int[],int[],int,int,int,int,int,Lokad.Onnx.DenseTensor`1[float])]
 
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
       FF157E26CCFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       48898568FFFFFF       mov      gword ptr [rbp-0x98], rax
       48899570FFFFFF       mov      qword ptr [rbp-0x90], rdx
       488DBD68FFFFFF       lea      rdi, [rbp-0x98]
       488D75B8             lea      rsi, [rbp-0x48]
       FF15B7AEB8FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG114:                ;; offset=0x09BA
       488BFB               mov      rdi, rbx
       FF155526CCFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       48898568FFFFFF       mov      gword ptr [rbp-0x98], rax
       48899570FFFFFF       mov      qword ptr [rbp-0x90], rdx
       488DBD68FFFFFF       lea      rdi, [rbp-0x98]
       488D75A0             lea      rsi, [rbp-0x60]
       FF158EAEB8FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG115:                ;; offset=0x09E3
       498BFE               mov      rdi, r14
       393F                 cmp      dword ptr [rdi], edi
       FF152A26CCFF         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       48898568FFFFFF       mov      gword ptr [rbp-0x98], rax
       48899570FFFFFF       mov      qword ptr [rbp-0x90], rdx
       488DBD68FFFFFF       lea      rdi, [rbp-0x98]
       488D7588             lea      rsi, [rbp-0x78]
       FF1563AEB8FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
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
       0F8F41060000         jg       G_M000_IG160
 
G_M000_IG117:                ;; offset=0x0A3C
       488B7020             mov      rsi, qword ptr [rax+0x20]
       4889B560FFFFFF       mov      qword ptr [rbp-0xA0], rsi
       488B7028             mov      rsi, qword ptr [rax+0x28]
       4889B558FFFFFF       mov      qword ptr [rbp-0xA8], rsi
       488B7030             mov      rsi, qword ptr [rax+0x30]
       4889B550FFFFFF       mov      qword ptr [rbp-0xB0], rsi
       4489A54CFFFFFF       mov      dword ptr [rbp-0xB4], r12d
       8BB54CFFFFFF         mov      esi, dword ptr [rbp-0xB4]
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E8B7F3F57C           call     CORINFO_HELP_NEWARR_1_VC
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
       E9BD0C0000           jmp      G_M000_IG190
 
G_M000_IG121:                ;; offset=0x0AB0
       E8C3180000           call     G_M000_IG362
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
       0F84600B0000         je       G_M000_IG184
 
G_M000_IG123:                ;; offset=0x0AE1
       4533C9               xor      r9d, r9d
 
G_M000_IG124:                ;; offset=0x0AE4
       8B8D94FEFFFF         mov      ecx, dword ptr [rbp-0x16C]
       3B8D18FFFFFF         cmp      ecx, dword ptr [rbp-0xE8]
       0F85A40B0000         jne      G_M000_IG186
 
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
       0F83400C0000         jae      G_M000_IG189
       8BF7                 mov      esi, edi
       488D54B110           lea      rdx, bword ptr [rcx+4*rsi+0x10]
       8B02                 mov      eax, dword ptr [rdx]
       FFC0                 inc      eax
       8902                 mov      dword ptr [rdx], eax
       4C8B8518FEFFFF       mov      r8, gword ptr [rbp-0x1E8]
       413B7808             cmp      edi, dword ptr [r8+0x08]
       0F83220C0000         jae      G_M000_IG189
       418B54B010           mov      edx, dword ptr [r8+4*rsi+0x10]
       039548FFFFFF         add      edx, dword ptr [rbp-0xB8]
       899548FFFFFF         mov      dword ptr [rbp-0xB8], edx
       4C8B8D10FEFFFF       mov      r9, gword ptr [rbp-0x1F0]
       413B7908             cmp      edi, dword ptr [r9+0x08]
       0F83000C0000         jae      G_M000_IG189
       418B54B110           mov      edx, dword ptr [r9+4*rsi+0x10]
       039544FFFFFF         add      edx, dword ptr [rbp-0xBC]
       899544FFFFFF         mov      dword ptr [rbp-0xBC], edx
       4C8B9508FEFFFF       mov      r10, gword ptr [rbp-0x1F8]
       413B7A08             cmp      edi, dword ptr [r10+0x08]
       0F83DE0B0000         jae      G_M000_IG189
       418B54B210           mov      edx, dword ptr [r10+4*rsi+0x10]
       039540FFFFFF         add      edx, dword ptr [rbp-0xC0]
       899540FFFFFF         mov      dword ptr [rbp-0xC0], edx
       3BFB                 cmp      edi, ebx
       0F83C50B0000         jae      G_M000_IG189
       3BFB                 cmp      edi, ebx
       0F83BD0B0000         jae      G_M000_IG189
       488B9520FEFFFF       mov      rdx, gword ptr [rbp-0x1E0]
       3B44B210             cmp      eax, dword ptr [rdx+4*rsi+0x10]
       0F8D39040000         jge      G_M000_IG159
 
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
       0F8D720B0000         jge      G_M000_IG190
 
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
       0F8D09050000         jge      G_M000_IG162
 
G_M000_IG132:                ;; offset=0x0C9E
       C5FE6F85C8FEFFFF     vmovdqu  ymm0, ymmword ptr [rbp-0x138]
       C5FE7F8598FEFFFF     vmovdqu  ymmword ptr [rbp-0x168], ymm0
       C5FA6F85E8FEFFFF     vmovdqu  xmm0, xmmword ptr [rbp-0x118]
       C5FA7F85B8FEFFFF     vmovdqu  xmmword ptr [rbp-0x148], xmm0
 
G_M000_IG133:                ;; offset=0x0CBE
       4533C9               xor      r9, r9
       4C898D88FEFFFF       mov      gword ptr [rbp-0x178], r9
       85DB                 test     ebx, ebx
       0F84F2020000         je       G_M000_IG158
 
G_M000_IG134:                ;; offset=0x0CD0
       4585FF               test     r15d, r15d
       740D                 je       SHORT G_M000_IG135
       83BD18FFFFFF01       cmp      dword ptr [rbp-0xE8], 1
       0F84B7050000         je       G_M000_IG163
 
G_M000_IG135:                ;; offset=0x0CE2
       4585FF               test     r15d, r15d
       0F84130A0000         je       G_M000_IG187
       83BD18FFFFFF02       cmp      dword ptr [rbp-0xE8], 2
       0F8C060A0000         jl       G_M000_IG187
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
       0F8493050000         je       G_M000_IG164
 
G_M000_IG136:                ;; offset=0x0D47
       83BD94FEFFFF40       cmp      dword ptr [rbp-0x16C], 64
       0F8DF4060000         jge      G_M000_IG172
 
G_M000_IG137:                ;; offset=0x0D54
       4585F6               test     r14d, r14d
       740D                 je       SHORT G_M000_IG139
 
G_M000_IG138:                ;; offset=0x0D59
       F68518FFFFFF01       test     byte  ptr [rbp-0xE8], 1
       0F855F080000         jne      G_M000_IG181
 
G_M000_IG139:                ;; offset=0x0D66
       4863BD14FFFFFF       movsxd   rdi, dword ptr [rbp-0xEC]
       48638510FFFFFF       movsxd   rax, dword ptr [rbp-0xF0]
       480FAFF8             imul     rdi, rax
       4881FF00000100       cmp      rdi, 0x10000
       0F8F40080000         jg       G_M000_IG181
 
G_M000_IG140:                ;; offset=0x0D85
       448BBD14FFFFFF       mov      r15d, dword ptr [rbp-0xEC]
       440FAFBD10FFFFFF     imul     r15d, dword ptr [rbp-0xF0]
       48BF682E80818C7C0000 mov      rdi, 0x7C8C81802E68
       488B3F               mov      rdi, gword ptr [rdi]
       418BF7               mov      esi, r15d
       FF15EED3D5FF         call     [System.Buffers.SharedArrayPool`1[float]:Rent(int):float[]:this]
       488BD8               mov      rbx, rax
       4D85E4               test     r12, r12
       7424                 je       SHORT G_M000_IG141
       48BEA069516B947C0000 mov      rsi, 0x7C946B5169A0
       49393424             cmp      qword ptr [r12], rsi
       0F85E3070000         jne      G_M000_IG180
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
       0F8407010000         je       G_M000_IG150
 
G_M000_IG143:                ;; offset=0x0DED
       837B0800             cmp      dword ptr [rbx+0x08], 0
       0F84FD000000         je       G_M000_IG150
       837B0800             cmp      dword ptr [rbx+0x08], 0
       0F86AD010000         jbe      G_M000_IG156
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
       7E6E                 jle      SHORT G_M000_IG154
 
G_M000_IG149:                ;; offset=0x0EEB
       4863FF               movsxd   rdi, edi
       48C1E702             shl      rdi, 2
       EB1E                 jmp      SHORT G_M000_IG152
 
G_M000_IG150:                ;; offset=0x0EF4
       4533C0               xor      r8d, r8d
       E90CFFFFFF           jmp      G_M000_IG144
       0F1F40000F1F840000000000 align    [12 bytes for IG153]
 
G_M000_IG151:                ;; offset=0x0F08
       FFC1                 inc      ecx
       3B8D14FFFFFF         cmp      ecx, dword ptr [rbp-0xEC]
       7D47                 jge      SHORT G_M000_IG154
 
G_M000_IG152:                ;; offset=0x0F12
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
       7EC8                 jle      SHORT G_M000_IG151
 
G_M000_IG153:                ;; offset=0x0F40
       4D63DA               movsxd   r11, r10d
       C4817A100499         vmovss   xmm0, dword ptr [r9+4*r11]
       C4A17A110498         vmovss   dword ptr [rax+4*r11], xmm0
       41FFC2               inc      r10d
       443BD6               cmp      r10d, esi
       7CE9                 jl       SHORT G_M000_IG153
       EBAF                 jmp      SHORT G_M000_IG151
 
G_M000_IG154:                ;; offset=0x0F59
       4585F6               test     r14d, r14d
       7428                 je       SHORT G_M000_IG155
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF152C25CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       EB2E                 jmp      SHORT G_M000_IG157
 
G_M000_IG155:                ;; offset=0x0F86
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF15EC24CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       EB06                 jmp      SHORT G_M000_IG157
 
G_M000_IG156:                ;; offset=0x0FAE
       E89D5B9AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG157:                ;; offset=0x0FB4
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
       E9EEFAFFFF           jmp      G_M000_IG121
 
G_M000_IG158:                ;; offset=0x0FC2
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF15F92ECCFF         call     [Lokad.Onnx.MathOps:mm(int,int,int,ptr,ptr,ptr)]
       E902FBFFFF           jmp      G_M000_IG125
 
G_M000_IG159:                ;; offset=0x0FF4
       3BFB                 cmp      edi, ebx
       0F836B070000         jae      G_M000_IG189
       33C0                 xor      eax, eax
       8944B110             mov      dword ptr [rcx+4*rsi+0x10], eax
       413B7808             cmp      edi, dword ptr [r8+0x08]
       0F835B070000         jae      G_M000_IG189
       418B44B010           mov      eax, dword ptr [r8+4*rsi+0x10]
       3BFB                 cmp      edi, ebx
       0F834E070000         jae      G_M000_IG189
       448B5CB210           mov      r11d, dword ptr [rdx+4*rsi+0x10]
       410FAFC3             imul     eax, r11d
       8B9D48FFFFFF         mov      ebx, dword ptr [rbp-0xB8]
       2BD8                 sub      ebx, eax
       899D48FFFFFF         mov      dword ptr [rbp-0xB8], ebx
       413B7908             cmp      edi, dword ptr [r9+0x08]
       0F832D070000         jae      G_M000_IG189
       418BC3               mov      eax, r11d
       410FAF44B110         imul     eax, dword ptr [r9+4*rsi+0x10]
       8B9D44FFFFFF         mov      ebx, dword ptr [rbp-0xBC]
       2BD8                 sub      ebx, eax
       899D44FFFFFF         mov      dword ptr [rbp-0xBC], ebx
       413B7A08             cmp      edi, dword ptr [r10+0x08]
       0F830C070000         jae      G_M000_IG189
       450FAF5CB210         imul     r11d, dword ptr [r10+4*rsi+0x10]
       8BB540FFFFFF         mov      esi, dword ptr [rbp-0xC0]
       412BF3               sub      esi, r11d
       89B540FFFFFF         mov      dword ptr [rbp-0xC0], esi
       FFCF                 dec      edi
       0F899BFAFFFF         jns      G_M000_IG127
       E93EFBFFFF           jmp      G_M000_IG128
 
G_M000_IG160:                ;; offset=0x107D
       448BBD1CFFFFFF       mov      r15d, dword ptr [rbp-0xE4]
       4963F7               movsxd   rsi, r15d
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E89AEDF57C           call     CORINFO_HELP_NEWARR_1_VC
       488B8D28FEFFFF       mov      rcx, gword ptr [rbp-0x1D8]
       488D7908             lea      rdi, bword ptr [rcx+0x08]
       488BF0               mov      rsi, rax
       E877439AFD           call     CORINFO_HELP_ASSIGN_REF
       4963F7               movsxd   rsi, r15d
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E875EDF57C           call     CORINFO_HELP_NEWARR_1_VC
       488B8D28FEFFFF       mov      rcx, gword ptr [rbp-0x1D8]
       488D7910             lea      rdi, bword ptr [rcx+0x10]
       488BF0               mov      rsi, rax
       E852439AFD           call     CORINFO_HELP_ASSIGN_REF
       4963F7               movsxd   rsi, r15d
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E850EDF57C           call     CORINFO_HELP_NEWARR_1_VC
       488B8D28FEFFFF       mov      rcx, gword ptr [rbp-0x1D8]
       488D7918             lea      rdi, bword ptr [rcx+0x18]
       488BF0               mov      rsi, rax
       E82D439AFD           call     CORINFO_HELP_ASSIGN_REF
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
       FF154022CCFF         call     [Lokad.Onnx.Tensor`1[float]:FillBatchOffsets(System.ReadOnlySpan`1[int],int[],int[],int[],int[],int[],int[])]
       48BF70105F6B947C0000 mov      rdi, 0x7C946B5F1070
       E8D9EBF57C           call     CORINFO_HELP_NEWSFAST
       4C8BF0               mov      r14, rax
       498BFE               mov      rdi, r14
 
G_M000_IG161:                ;; offset=0x114D
       FF153D22CCFF         call     [System.Threading.Tasks.ParallelOptions:.ctor():this]
       498BFE               mov      rdi, r14
       8BF3                 mov      esi, ebx
       FF154A22CCFF         call     [System.Threading.Tasks.ParallelOptions:set_MaxDegreeOfParallelism(int):this]
       48BF98115F6B947C0000 mov      rdi, 0x7C946B5F1198
       E8B3EBF57C           call     CORINFO_HELP_NEWSFAST
       488BD8               mov      rbx, rax
       488BFB               mov      rdi, rbx
       488BB528FEFFFF       mov      rsi, gword ptr [rbp-0x1D8]
       48BAD89E556B947C0000 mov      rdx, 0x7C946B559ED8
       FF155EB09AFE         call     [System.MulticastDelegate:CtorClosed(System.Object,nint):this]
       488DBD20FFFFFF       lea      rdi, [rbp-0xE0]
       4C8BC3               mov      r8, rbx
       418BD7               mov      edx, r15d
       498BCE               mov      rcx, r14
       33F6                 xor      esi, esi
       FF151E22CCFF         call     [System.Threading.Tasks.Parallel:For(int,int,System.Threading.Tasks.ParallelOptions,System.Action`1[int]):System.Threading.Tasks.ParallelLoopResult]
       E9C6050000           jmp      G_M000_IG190
 
G_M000_IG162:                ;; offset=0x11A7
       81BD14FFFFFF00040000 cmp      dword ptr [rbp-0xEC], 0x400
       0F8CE7FAFFFF         jl       G_M000_IG132
       81BD10FFFFFF00040000 cmp      dword ptr [rbp-0xF0], 0x400
       0F8CD7FAFFFF         jl       G_M000_IG132
       48638D14FFFFFF       movsxd   rcx, dword ptr [rbp-0xEC]
       4863BD10FFFFFF       movsxd   rdi, dword ptr [rbp-0xF0]
       480FAFCF             imul     rcx, rdi
       4881F900000004       cmp      rcx, 0x4000000
       0F8FB8FAFFFF         jg       G_M000_IG132
       85DB                 test     ebx, ebx
       0F84B0FAFFFF         je       G_M000_IG132
       4585FF               test     r15d, r15d
       0F84A7FAFFFF         je       G_M000_IG132
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
       FF157C2BCCFF         call     [Lokad.Onnx.Tensor`1[float]:RunIsolatedShortWidePackedRows(int,int,int,ptr,ptr,ptr,Lokad.Onnx.TensorExecutionOptions)]
       E966F8FFFF           jmp      G_M000_IG126
 
G_M000_IG163:                ;; offset=0x1299
       81BD10FFFFFF00200000 cmp      dword ptr [rbp-0xF0], 0x2000
       0F8C39FAFFFF         jl       G_M000_IG135
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       BF01000000           mov      edi, 1
       FF159B2BCCFF         call     [Lokad.Onnx.MathOps:mm_m1_kblocked(int,int,int,ptr,ptr,ptr)]
       E91CF8FFFF           jmp      G_M000_IG125
 
G_M000_IG164:                ;; offset=0x12DA
       83BD18FFFFFF40       cmp      dword ptr [rbp-0xE8], 64
       0F8C60FAFFFF         jl       G_M000_IG136
       48638D14FFFFFF       movsxd   rcx, dword ptr [rbp-0xEC]
       4863BD10FFFFFF       movsxd   rdi, dword ptr [rbp-0xF0]
       480FAFCF             imul     rcx, rdi
       4881F900000004       cmp      rcx, 0x4000000
       0F8F41FAFFFF         jg       G_M000_IG136
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
       FF15B72ACCFF         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488985C0FDFFFF       mov      gword ptr [rbp-0x240], rax
 
G_M000_IG165:                ;; offset=0x1390
       488BBDC0FDFFFF       mov      rdi, gword ptr [rbp-0x240]
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
       4883BDC0FDFFFF00     cmp      gword ptr [rbp-0x240], 0
       740D                 je       SHORT G_M000_IG166
       488BBDC0FDFFFF       mov      rdi, gword ptr [rbp-0x240]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       750B                 jne      SHORT G_M000_IG168
 
G_M000_IG166:                ;; offset=0x13B5
       4533E4               xor      r12d, r12d
       EB1E                 jmp      SHORT G_M000_IG169
 
G_M000_IG167:                ;; offset=0x13BA
       E891579AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG168:                ;; offset=0x13C0
       488BBDC0FDFFFF       mov      rdi, gword ptr [rbp-0x240]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       76ED                 jbe      SHORT G_M000_IG167
       4C8BA5C0FDFFFF       mov      r12, gword ptr [rbp-0x240]
       4983C410             add      r12, 16
 
G_M000_IG169:                ;; offset=0x13D8
       8BBD14FFFFFF         mov      edi, dword ptr [rbp-0xEC]
       8BB510FFFFFF         mov      esi, dword ptr [rbp-0xF0]
       488B9500FFFFFF       mov      rdx, qword ptr [rbp-0x100]
       498BCC               mov      rcx, r12
       FF15ACA4B8FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4D8BC4               mov      r8, r12
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF157B20CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG170:                ;; offset=0x141E
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG171:                ;; offset=0x1427
       48BF682E80818C7C0000 mov      rdi, 0x7C8C81802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB5C0FDFFFF       mov      rsi, gword ptr [rbp-0x240]
       33D2                 xor      edx, edx
       FF155DCDD5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       E9F9010000           jmp      G_M000_IG184
 
G_M000_IG172:                ;; offset=0x1448
       48638D14FFFFFF       movsxd   rcx, dword ptr [rbp-0xEC]
       4863BD10FFFFFF       movsxd   rdi, dword ptr [rbp-0xF0]
       480FAFCF             imul     rcx, rdi
       4881F900000004       cmp      rcx, 0x4000000
       0F8FEDF8FFFF         jg       G_M000_IG137
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
       FF155629CCFF         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       488985C8FDFFFF       mov      gword ptr [rbp-0x238], rax
 
G_M000_IG173:                ;; offset=0x14F1
       488BBDC8FDFFFF       mov      rdi, gword ptr [rbp-0x238]
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
       4883BDC8FDFFFF00     cmp      gword ptr [rbp-0x238], 0
       740D                 je       SHORT G_M000_IG174
       488BBDC8FDFFFF       mov      rdi, gword ptr [rbp-0x238]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       750B                 jne      SHORT G_M000_IG176
 
G_M000_IG174:                ;; offset=0x1516
       4533F6               xor      r14d, r14d
       EB1E                 jmp      SHORT G_M000_IG177
 
G_M000_IG175:                ;; offset=0x151B
       E830569AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG176:                ;; offset=0x1521
       488BBDC8FDFFFF       mov      rdi, gword ptr [rbp-0x238]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       76ED                 jbe      SHORT G_M000_IG175
       4C8BB5C8FDFFFF       mov      r14, gword ptr [rbp-0x238]
       4983C610             add      r14, 16
 
G_M000_IG177:                ;; offset=0x1539
       8BBD14FFFFFF         mov      edi, dword ptr [rbp-0xEC]
       8BB510FFFFFF         mov      esi, dword ptr [rbp-0xF0]
       488B9500FFFFFF       mov      rdx, qword ptr [rbp-0x100]
       498BCE               mov      rcx, r14
       FF154BA3B8FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8BBD94FEFFFF         mov      edi, dword ptr [rbp-0x16C]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4D8BC6               mov      r8, r14
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF15321FCCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG178:                ;; offset=0x157F
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG179:                ;; offset=0x1588
       48BF682E80818C7C0000 mov      rdi, 0x7C8C81802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB5C8FDFFFF       mov      rsi, gword ptr [rbp-0x238]
       33D2                 xor      edx, edx
       FF15FCCBD5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       E90DF5FFFF           jmp      G_M000_IG122
 
G_M000_IG180:                ;; offset=0x15A9
       4963F7               movsxd   rsi, r15d
       48C1E602             shl      rsi, 2
       498BFC               mov      rdi, r12
       49BBA0262669947C0000 mov      r11, 0x7C94692626A0
       41FF13               call     [r11]Lokad.Onnx.IScratchAccountant:AddScratchBytes(long):this
       E911F8FFFF           jmp      G_M000_IG141
 
G_M000_IG181:                ;; offset=0x15C5
       81BD14FFFFFF000A0000 cmp      dword ptr [rbp-0xEC], 0xA00
       7D3E                 jge      SHORT G_M000_IG183
 
G_M000_IG182:                ;; offset=0x15D1
       81BD10FFFFFF000A0000 cmp      dword ptr [rbp-0xF0], 0xA00
       7D32                 jge      SHORT G_M000_IG183
       8BBD94FEFFFF         mov      edi, dword ptr [rbp-0x16C]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF157E28CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4tiled(int,int,int,ptr,ptr,ptr)]
       E9A7F4FFFF           jmp      G_M000_IG122
 
G_M000_IG183:                ;; offset=0x160F
       8BBD94FEFFFF         mov      edi, dword ptr [rbp-0x16C]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF156428CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4(int,int,int,ptr,ptr,ptr)]
       E975F4FFFF           jmp      G_M000_IG122
 
G_M000_IG184:                ;; offset=0x1641
       4C638D14FFFFFF       movsxd   r9, dword ptr [rbp-0xEC]
       48638D10FFFFFF       movsxd   rcx, dword ptr [rbp-0xF0]
       4C0FAFC9             imul     r9, rcx
       4981F900000004       cmp      r9, 0x4000000
       0F8F81F4FFFF         jg       G_M000_IG123
       83BD18FFFFFF40       cmp      dword ptr [rbp-0xE8], 64
       7D26                 jge      SHORT G_M000_IG185
       4C638D14FFFFFF       movsxd   r9, dword ptr [rbp-0xEC]
       48638D10FFFFFF       movsxd   rcx, dword ptr [rbp-0xF0]
       4C0FAFC9             imul     r9, rcx
       4981F900000100       cmp      r9, 0x10000
       410F9EC1             setle    r9b
       450FB6C9             movzx    r9, r9b
       E955F4FFFF           jmp      G_M000_IG124
 
G_M000_IG185:                ;; offset=0x168F
       41B901000000         mov      r9d, 1
       E94AF4FFFF           jmp      G_M000_IG124
 
G_M000_IG186:                ;; offset=0x169A
       4585C9               test     r9d, r9d
       0F8553F4FFFF         jne      G_M000_IG125
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
       FF15BF27CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E9F8F3FFFF           jmp      G_M000_IG125
 
G_M000_IG187:                ;; offset=0x16FE
       4585FF               test     r15d, r15d
       7432                 je       SHORT G_M000_IG188
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF158827CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E9C1F3FFFF           jmp      G_M000_IG125
 
G_M000_IG188:                ;; offset=0x1735
       8BBD18FFFFFF         mov      edi, dword ptr [rbp-0xE8]
       8BB514FFFFFF         mov      esi, dword ptr [rbp-0xEC]
       8B9510FFFFFF         mov      edx, dword ptr [rbp-0xF0]
       488B8D08FFFFFF       mov      rcx, qword ptr [rbp-0xF8]
       4C8B8500FFFFFF       mov      r8, qword ptr [rbp-0x100]
       4C8B8DF8FEFFFF       mov      r9, qword ptr [rbp-0x108]
       FF156E27CCFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized(int,int,int,ptr,ptr,ptr)]
       E98FF3FFFF           jmp      G_M000_IG125
 
G_M000_IG189:                ;; offset=0x1767
       E8E4539AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG190:                ;; offset=0x176D
       48837D9800           cmp      qword ptr [rbp-0x68], 0
       740A                 je       SHORT G_M000_IG192
 
G_M000_IG191:                ;; offset=0x1774
       488D7D98             lea      rdi, [rbp-0x68]
       FF15521EA1FF         call     [System.Runtime.InteropServices.GCHandle:Free():this]
 
G_M000_IG192:                ;; offset=0x177E
       48837D8800           cmp      gword ptr [rbp-0x78], 0
       7508                 jne      SHORT G_M000_IG194
 
G_M000_IG193:                ;; offset=0x1785
       33FF                 xor      edi, edi
       48897D90             mov      qword ptr [rbp-0x70], rdi
       EB19                 jmp      SHORT G_M000_IG195
 
G_M000_IG194:                ;; offset=0x178D
       488B7D88             mov      rdi, gword ptr [rbp-0x78]
       49BBA8262669947C0000 mov      r11, 0x7C94692626A8
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33FF                 xor      rdi, rdi
       48897D88             mov      gword ptr [rbp-0x78], rdi
       EBDF                 jmp      SHORT G_M000_IG193
 
G_M000_IG195:                ;; offset=0x17A6
       48837DB000           cmp      qword ptr [rbp-0x50], 0
       740A                 je       SHORT G_M000_IG197
 
G_M000_IG196:                ;; offset=0x17AD
       488D7DB0             lea      rdi, [rbp-0x50]
       FF15191EA1FF         call     [System.Runtime.InteropServices.GCHandle:Free():this]
 
G_M000_IG197:                ;; offset=0x17B7
       48837DA000           cmp      gword ptr [rbp-0x60], 0
       7508                 jne      SHORT G_M000_IG199
 
G_M000_IG198:                ;; offset=0x17BE
       33FF                 xor      edi, edi
       48897DA8             mov      qword ptr [rbp-0x58], rdi
       EB19                 jmp      SHORT G_M000_IG200
 
G_M000_IG199:                ;; offset=0x17C6
       488B7DA0             mov      rdi, gword ptr [rbp-0x60]
       49BBB0262669947C0000 mov      r11, 0x7C94692626B0
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
       EBDF                 jmp      SHORT G_M000_IG198
 
G_M000_IG200:                ;; offset=0x17DF
       48837DC800           cmp      qword ptr [rbp-0x38], 0
       740A                 je       SHORT G_M000_IG202
 
G_M000_IG201:                ;; offset=0x17E6
       488D7DC8             lea      rdi, [rbp-0x38]
       FF15E01DA1FF         call     [System.Runtime.InteropServices.GCHandle:Free():this]
 
G_M000_IG202:                ;; offset=0x17F0
       48837DB800           cmp      gword ptr [rbp-0x48], 0
       0F85CA0A0000         jne      G_M000_IG345
 
G_M000_IG203:                ;; offset=0x17FB
       33C0                 xor      eax, eax
       488945C0             mov      qword ptr [rbp-0x40], rax
       E970F1FFFF           jmp      G_M000_IG110
 
G_M000_IG204:                ;; offset=0x1806
       4883F820             cmp      rax, 32
       736B                 jae      SHORT G_M000_IG212
 
G_M000_IG205:                ;; offset=0x180C
       4883F810             cmp      rax, 16
       7322                 jae      SHORT G_M000_IG208
 
G_M000_IG206:                ;; offset=0x1812
       4883C0F8             add      rax, -8
       488B11               mov      rdx, qword ptr [rcx]
       482B17               sub      rdx, qword ptr [rdi]
       488B3401             mov      rsi, qword ptr [rcx+rax]
       482B3407             sub      rsi, qword ptr [rdi+rax]
       480BD6               or       rdx, rsi
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
 
G_M000_IG207:                ;; offset=0x182D
       8BF8                 mov      edi, eax
       E997E9FFFF           jmp      G_M000_IG18
 
G_M000_IG208:                ;; offset=0x1834
       33C9                 xor      ecx, ecx
       4883C0F0             add      rax, -16
       7519                 jne      SHORT G_M000_IG210
 
G_M000_IG209:                ;; offset=0x183C
       C5F8100402           vmovups  xmm0, xmmword ptr [rdx+rax]
       62F17D08740C06       vpcmpeqb k1, xmm0, xmmword ptr [rsi+rax]
       C5F898C9             kortestw k1, k1
       736B                 jae      SHORT G_M000_IG216
       E973E9FFFF           jmp      G_M000_IG17
 
G_M000_IG210:                ;; offset=0x1853
       C5F81002             vmovups  xmm0, xmmword ptr [rdx]
       62F37D083E0E04       vpcmpfalseub k1, xmm0, xmmword ptr [rsi]
       C5F898C9             kortestw k1, k1
       7553                 jne      SHORT G_M000_IG216
 
G_M000_IG211:                ;; offset=0x1864
       4883C110             add      rcx, 16
       483BC1               cmp      rax, rcx
       0F87DD030000         ja       G_M000_IG273
       EBC9                 jmp      SHORT G_M000_IG209
 
G_M000_IG212:                ;; offset=0x1873
       33FF                 xor      edi, edi
       4883C0E0             add      rax, -32
       7422                 je       SHORT G_M000_IG215
 
G_M000_IG213:                ;; offset=0x187B
       C5FC1002             vmovups  ymm0, ymmword ptr [rdx]
       62F37D283E0E04       vpcmpfalseub k1, ymm0, ymmword ptr [rsi]
       C4E1F998C9           kortestd k1, k1
       7527                 jne      SHORT G_M000_IG216
 
G_M000_IG214:                ;; offset=0x188D
       4883C720             add      rdi, 32
       483BC7               cmp      rax, rdi
       0F8791030000         ja       G_M000_IG272
 
G_M000_IG215:                ;; offset=0x189A
       C5FC100402           vmovups  ymm0, ymmword ptr [rdx+rax]
       62F17D28740C06       vpcmpeqb k1, ymm0, ymmword ptr [rsi+rax]
       C4E1F998C9           kortestd k1, k1
       0F8215E9FFFF         jb       G_M000_IG17
 
G_M000_IG216:                ;; offset=0x18B1
       33FF                 xor      edi, edi
       E913E9FFFF           jmp      G_M000_IG18
 
G_M000_IG217:                ;; offset=0x18B8
       4883F820             cmp      rax, 32
       736B                 jae      SHORT G_M000_IG225
 
G_M000_IG218:                ;; offset=0x18BE
       4883F810             cmp      rax, 16
       7322                 jae      SHORT G_M000_IG221
 
G_M000_IG219:                ;; offset=0x18C4
       4883C0F8             add      rax, -8
       488B11               mov      rdx, qword ptr [rcx]
       482B17               sub      rdx, qword ptr [rdi]
       488B3401             mov      rsi, qword ptr [rcx+rax]
       482B3407             sub      rsi, qword ptr [rdi+rax]
       480BD6               or       rdx, rsi
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
 
G_M000_IG220:                ;; offset=0x18DF
       8BF8                 mov      edi, eax
       E941EAFFFF           jmp      G_M000_IG36
 
G_M000_IG221:                ;; offset=0x18E6
       33C9                 xor      ecx, ecx
       4883C0F0             add      rax, -16
       7519                 jne      SHORT G_M000_IG223
 
G_M000_IG222:                ;; offset=0x18EE
       C5F8100402           vmovups  xmm0, xmmword ptr [rdx+rax]
       62F17D08740C06       vpcmpeqb k1, xmm0, xmmword ptr [rsi+rax]
       C5F898C9             kortestw k1, k1
       736B                 jae      SHORT G_M000_IG229
       E91DEAFFFF           jmp      G_M000_IG35
 
G_M000_IG223:                ;; offset=0x1905
       C5F81002             vmovups  xmm0, xmmword ptr [rdx]
       62F37D083E0E04       vpcmpfalseub k1, xmm0, xmmword ptr [rsi]
       C5F898C9             kortestw k1, k1
       7553                 jne      SHORT G_M000_IG229
 
G_M000_IG224:                ;; offset=0x1916
       4883C110             add      rcx, 16
       483BC1               cmp      rax, rcx
       0F87E0040000         ja       G_M000_IG292
       EBC9                 jmp      SHORT G_M000_IG222
 
G_M000_IG225:                ;; offset=0x1925
       33FF                 xor      edi, edi
       4883C0E0             add      rax, -32
       7422                 je       SHORT G_M000_IG228
 
G_M000_IG226:                ;; offset=0x192D
       C5FC1002             vmovups  ymm0, ymmword ptr [rdx]
       62F37D283E0E04       vpcmpfalseub k1, ymm0, ymmword ptr [rsi]
       C4E1F998C9           kortestd k1, k1
       7527                 jne      SHORT G_M000_IG229
 
G_M000_IG227:                ;; offset=0x193F
       4883C720             add      rdi, 32
       483BC7               cmp      rax, rdi
       0F8794040000         ja       G_M000_IG291
 
G_M000_IG228:                ;; offset=0x194C
       C5FC100402           vmovups  ymm0, ymmword ptr [rdx+rax]
       62F17D28740C06       vpcmpeqb k1, ymm0, ymmword ptr [rsi+rax]
       C4E1F998C9           kortestd k1, k1
       0F82BFE9FFFF         jb       G_M000_IG35
 
G_M000_IG229:                ;; offset=0x1963
       33FF                 xor      edi, edi
       E9BDE9FFFF           jmp      G_M000_IG36
 
G_M000_IG230:                ;; offset=0x196A
       4883F820             cmp      rax, 32
       736B                 jae      SHORT G_M000_IG238
 
G_M000_IG231:                ;; offset=0x1970
       4883F810             cmp      rax, 16
       7322                 jae      SHORT G_M000_IG234
 
G_M000_IG232:                ;; offset=0x1976
       4883C0F8             add      rax, -8
       488B11               mov      rdx, qword ptr [rcx]
       482B17               sub      rdx, qword ptr [rdi]
       488B3401             mov      rsi, qword ptr [rcx+rax]
       482B3407             sub      rsi, qword ptr [rdi+rax]
       480BD6               or       rdx, rsi
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
 
G_M000_IG233:                ;; offset=0x1991
       8BF8                 mov      edi, eax
       E9C5EAFFFF           jmp      G_M000_IG51
 
G_M000_IG234:                ;; offset=0x1998
       33C9                 xor      ecx, ecx
       4883C0F0             add      rax, -16
       7519                 jne      SHORT G_M000_IG236
 
G_M000_IG235:                ;; offset=0x19A0
       C5F8100402           vmovups  xmm0, xmmword ptr [rdx+rax]
       62F17D08740C06       vpcmpeqb k1, xmm0, xmmword ptr [rsi+rax]
       C5F898C9             kortestw k1, k1
       736B                 jae      SHORT G_M000_IG242
       E9A1EAFFFF           jmp      G_M000_IG50
 
G_M000_IG236:                ;; offset=0x19B7
       C5F81002             vmovups  xmm0, xmmword ptr [rdx]
       62F37D083E0E04       vpcmpfalseub k1, xmm0, xmmword ptr [rsi]
       C5F898C9             kortestw k1, k1
       7553                 jne      SHORT G_M000_IG242
 
G_M000_IG237:                ;; offset=0x19C8
       4883C110             add      rcx, 16
       483BC1               cmp      rax, rcx
       0F87B6050000         ja       G_M000_IG309
       EBC9                 jmp      SHORT G_M000_IG235
 
G_M000_IG238:                ;; offset=0x19D7
       33FF                 xor      edi, edi
       4883C0E0             add      rax, -32
       7422                 je       SHORT G_M000_IG241
 
G_M000_IG239:                ;; offset=0x19DF
       C5FC1002             vmovups  ymm0, ymmword ptr [rdx]
       62F37D283E0E04       vpcmpfalseub k1, ymm0, ymmword ptr [rsi]
       C4E1F998C9           kortestd k1, k1
       7527                 jne      SHORT G_M000_IG242
 
G_M000_IG240:                ;; offset=0x19F1
       4883C720             add      rdi, 32
       483BC7               cmp      rax, rdi
       0F876A050000         ja       G_M000_IG308
 
G_M000_IG241:                ;; offset=0x19FE
       C5FC100402           vmovups  ymm0, ymmword ptr [rdx+rax]
       62F17D28740C06       vpcmpeqb k1, ymm0, ymmword ptr [rsi+rax]
       C4E1F998C9           kortestd k1, k1
       0F8243EAFFFF         jb       G_M000_IG50
 
G_M000_IG242:                ;; offset=0x1A15
       33FF                 xor      edi, edi
       E941EAFFFF           jmp      G_M000_IG51
 
G_M000_IG243:                ;; offset=0x1A1C
       8BC6                 mov      eax, esi
       418B448210           mov      eax, dword ptr [r10+4*rax+0x10]
       E929ECFFFF           jmp      G_M000_IG70
 
G_M000_IG244:                ;; offset=0x1A28
       448B8D48FEFFFF       mov      r9d, dword ptr [rbp-0x1B8]
       458D41FE             lea      r8d, [r9-0x02]
       413BF0               cmp      esi, r8d
       7D1C                 jge      SHORT G_M000_IG246
 
G_M000_IG245:                ;; offset=0x1A38
       413BF1               cmp      esi, r9d
       0F8382080000         jae      G_M000_IG346
       8BC6                 mov      eax, esi
       488B9588FDFFFF       mov      rdx, bword ptr [rbp-0x278]
       833C8201             cmp      dword ptr [rdx+4*rax], 1
       0F8597060000         jne      G_M000_IG325
 
G_M000_IG246:                ;; offset=0x1A54
       33C0                 xor      eax, eax
 
G_M000_IG247:                ;; offset=0x1A56
       3B7108               cmp      esi, dword ptr [rcx+0x08]
       0F8364080000         jae      G_M000_IG346
       448BC6               mov      r8d, esi
       4289448110           mov      dword ptr [rcx+4*r8+0x10], eax
       FFC6                 inc      esi
       3BF7                 cmp      esi, edi
       7CBB                 jl       SHORT G_M000_IG244
       E9EDEBFFFF           jmp      G_M000_IG71
 
G_M000_IG248:                ;; offset=0x1A72
       8BCE                 mov      ecx, esi
       418B4C8910           mov      ecx, dword ptr [r9+4*rcx+0x10]
       E9B7ECFFFF           jmp      G_M000_IG84
 
G_M000_IG249:                ;; offset=0x1A7E
       448B9538FEFFFF       mov      r10d, dword ptr [rbp-0x1C8]
       418D52FE             lea      edx, [r10-0x02]
       3BF2                 cmp      esi, edx
       7D1D                 jge      SHORT G_M000_IG251
 
G_M000_IG250:                ;; offset=0x1A8D
       413BF2               cmp      esi, r10d
       0F832D080000         jae      G_M000_IG346
       8BCE                 mov      ecx, esi
       4C8B8578FDFFFF       mov      r8, bword ptr [rbp-0x288]
       41833C8801           cmp      dword ptr [r8+4*rcx], 1
       0F85D8060000         jne      G_M000_IG332
 
G_M000_IG251:                ;; offset=0x1AAA
       33C9                 xor      ecx, ecx
 
G_M000_IG252:                ;; offset=0x1AAC
       3B7708               cmp      esi, dword ptr [rdi+0x08]
       0F830E080000         jae      G_M000_IG346
       8BD6                 mov      edx, esi
       894C9710             mov      dword ptr [rdi+4*rdx+0x10], ecx
       FFC6                 inc      esi
       3BF0                 cmp      esi, eax
       7CBD                 jl       SHORT G_M000_IG249
       E97DECFFFF           jmp      G_M000_IG85
 
G_M000_IG253:                ;; offset=0x1AC6
       8BC8                 mov      ecx, eax
       8B4C8F10             mov      ecx, dword ptr [rdi+4*rcx+0x10]
       E91CEDFFFF           jmp      G_M000_IG95
 
G_M000_IG254:                ;; offset=0x1AD1
       448B9530FEFFFF       mov      r10d, dword ptr [rbp-0x1D0]
       418D72FE             lea      esi, [r10-0x02]
       3BC6                 cmp      eax, esi
       7D1D                 jge      SHORT G_M000_IG256
 
G_M000_IG255:                ;; offset=0x1AE0
       413BC2               cmp      eax, r10d
       0F83DA070000         jae      G_M000_IG346
       8BC8                 mov      ecx, eax
       4C8B8D70FDFFFF       mov      r9, bword ptr [rbp-0x290]
       41833C8901           cmp      dword ptr [r9+4*rcx], 1
       0F85B3060000         jne      G_M000_IG337
 
G_M000_IG256:                ;; offset=0x1AFD
       33C9                 xor      ecx, ecx
 
G_M000_IG257:                ;; offset=0x1AFF
       413B4008             cmp      eax, dword ptr [r8+0x08]
       0F83BA070000         jae      G_M000_IG346
       8BF0                 mov      esi, eax
       41894CB010           mov      dword ptr [r8+4*rsi+0x10], ecx
       FFC0                 inc      eax
       3BC2                 cmp      eax, edx
       7CBB                 jl       SHORT G_M000_IG254
       E9E0ECFFFF           jmp      G_M000_IG96
 
G_M000_IG258:                ;; offset=0x1B1B
       498BF7               mov      rsi, r15
       48BF301A5C6B947C0000 mov      rdi, 0x7C946B5C1A30
       E853A0EAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       488BF8               mov      rdi, rax
       E97DE5FFFF           jmp      G_M000_IG04
 
G_M000_IG259:                ;; offset=0x1B35
       FF15B518CCFF         call     [Lokad.Onnx.Tensor`1[float]:HasDenseMatrixCore(Lokad.Onnx.BroadcastedTensor`1[float]):bool]
       85C0                 test     eax, eax
       0F8478E5FFFF         je       G_M000_IG05
       E99EE6FFFF           jmp      G_M000_IG20
 
G_M000_IG260:                ;; offset=0x1B48
       498BF7               mov      rsi, r15
       E830A0EAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BE0               mov      r12, rax
       E97FE5FFFF           jmp      G_M000_IG07
 
G_M000_IG261:                ;; offset=0x1B58
       33C9                 xor      rcx, rcx
       48898DA8FDFFFF       mov      bword ptr [rbp-0x258], rcx
       4533C0               xor      r8d, r8d
       44898560FEFFFF       mov      dword ptr [rbp-0x1A0], r8d
       488B8DA8FDFFFF       mov      rcx, bword ptr [rbp-0x258]
       448B8560FEFFFF       mov      r8d, dword ptr [rbp-0x1A0]
       E984E5FFFF           jmp      G_M000_IG08
 
G_M000_IG262:                ;; offset=0x1B7E
       33F6                 xor      rsi, rsi
       33D2                 xor      edx, edx
       48898DA8FDFFFF       mov      bword ptr [rbp-0x258], rcx
       44898560FEFFFF       mov      dword ptr [rbp-0x1A0], r8d
       E990E5FFFF           jmp      G_M000_IG09
 
G_M000_IG263:                ;; offset=0x1B95
       33FF                 xor      rdi, rdi
       33C0                 xor      eax, eax
       E9A4E5FFFF           jmp      G_M000_IG10
 
G_M000_IG264:                ;; offset=0x1B9E
       4883F804             cmp      rax, 4
       7333                 jae      SHORT G_M000_IG270
 
G_M000_IG265:                ;; offset=0x1BA4
       33D2                 xor      edx, edx
       488BF0               mov      rsi, rax
       4883E602             and      rsi, 2
       740A                 je       SHORT G_M000_IG267
 
G_M000_IG266:                ;; offset=0x1BAF
       0FB711               movzx    rdx, word  ptr [rcx]
       440FB707             movzx    r8, word  ptr [rdi]
       412BD0               sub      edx, r8d
 
G_M000_IG267:                ;; offset=0x1BB9
       A801                 test     al, 1
       740D                 je       SHORT G_M000_IG269
 
G_M000_IG268:                ;; offset=0x1BBD
       0FB60431             movzx    rax, byte  ptr [rcx+rsi]
       400FB63C37           movzx    rdi, byte  ptr [rdi+rsi]
       2BC7                 sub      eax, edi
       0BD0                 or       edx, eax
 
G_M000_IG269:                ;; offset=0x1BCA
       85D2                 test     edx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E956FCFFFF           jmp      G_M000_IG207
 
G_M000_IG270:                ;; offset=0x1BD7
       4883C0FC             add      rax, -4
       8B11                 mov      edx, dword ptr [rcx]
       2B17                 sub      edx, dword ptr [rdi]
       8B0C01               mov      ecx, dword ptr [rcx+rax]
       2B0C07               sub      ecx, dword ptr [rdi+rax]
       0BCA                 or       ecx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E93BFCFFFF           jmp      G_M000_IG207
 
G_M000_IG271:                ;; offset=0x1BF2
       62F17C4810040A       vmovups  zmm0, zmmword ptr [rdx+rcx]
       62F37D483E0C0E04     vpcmpfalseub k1, zmm0, zmmword ptr [rsi+rcx]
       C4E1F898C9           kortestq k1, k1
       0F85A5FCFFFF         jne      G_M000_IG216
       E98CE5FFFF           jmp      G_M000_IG15
 
G_M000_IG272:                ;; offset=0x1C11
       C5FC10043A           vmovups  ymm0, ymmword ptr [rdx+rdi]
       62F37D283E0C3E04     vpcmpfalseub k1, ymm0, ymmword ptr [rsi+rdi]
       C4E1F998C9           kortestd k1, k1
       0F8588FCFFFF         jne      G_M000_IG216
       E95FFCFFFF           jmp      G_M000_IG214
 
G_M000_IG273:                ;; offset=0x1C2E
       C5F810040A           vmovups  xmm0, xmmword ptr [rdx+rcx]
       62F37D083E0C0E04     vpcmpfalseub k1, xmm0, xmmword ptr [rsi+rcx]
       C5F898C9             kortestw k1, k1
       0F856CFCFFFF         jne      G_M000_IG216
       E91AFCFFFF           jmp      G_M000_IG211
 
G_M000_IG274:                ;; offset=0x1C4A
       33FF                 xor      edi, edi
       E97AE5FFFF           jmp      G_M000_IG18
 
G_M000_IG275:                ;; offset=0x1C51
       48BF28CD286A947C0000 mov      rdi, 0x7C946A28CD28
       E8C0E0F57C           call     CORINFO_HELP_NEWSFAST
       4C8BF8               mov      r15, rax
       BF53E60000           mov      edi, 0xE653
       48BE087F346A947C0000 mov      rsi, 0x7C946A347F08
       FF15707B61FF         call     [CORINFO_HELP_STRCNS]
       4C8BE8               mov      r13, rax
       BF1FE90000           mov      edi, 0xE91F
       48BE087F346A947C0000 mov      rsi, 0x7C946A347F08
       FF15587B61FF         call     [CORINFO_HELP_STRCNS]
       488BF0               mov      rsi, rax
       498BFD               mov      rdi, r13
       FF15AC139BFE         call     [System.String:Concat(System.String,System.String):System.String]
       488BF0               mov      rsi, rax
       498BFF               mov      rdi, r15
       FF15A07B61FF         call     [System.ArgumentException:.ctor(System.String):this]
       498BFF               mov      rdi, r15
       E8100DE17C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG276:                ;; offset=0x1CB1
       498BFF               mov      rdi, r15
       498B07               mov      rax, qword ptr [r15]
       488B4078             mov      rax, qword ptr [rax+0x78]
       FF5010               call     [rax+0x10]Lokad.Onnx.Tensor`1[float]:ToDenseTensor():Lokad.Onnx.DenseTensor`1[float]:this
       488BF8               mov      rdi, rax
       498BF5               mov      rsi, r13
       FF159E05CCFF         call     [Lokad.Onnx.Tensor`1[float]:CountedCopy[float](Lokad.Onnx.DenseTensor`1[float],Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE0               mov      r12, rax
       E911E5FFFF           jmp      G_M000_IG19
 
G_M000_IG277:                ;; offset=0x1CD2
       488BF3               mov      rsi, rbx
       48BF301A5C6B947C0000 mov      rdi, 0x7C946B5C1A30
       E89C9EEAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       488BF8               mov      rdi, rax
       E922E5FFFF           jmp      G_M000_IG22
 
G_M000_IG278:                ;; offset=0x1CEC
       FF15FE16CCFF         call     [Lokad.Onnx.Tensor`1[float]:HasDenseMatrixCore(Lokad.Onnx.BroadcastedTensor`1[float]):bool]
       85C0                 test     eax, eax
       0F841DE5FFFF         je       G_M000_IG23
       E943E6FFFF           jmp      G_M000_IG38
 
G_M000_IG279:                ;; offset=0x1CFF
       488BF3               mov      rsi, rbx
       E8799EEAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BE0               mov      r12, rax
       E924E5FFFF           jmp      G_M000_IG25
 
G_M000_IG280:                ;; offset=0x1D0F
       33C9                 xor      rcx, rcx
       48898DA0FDFFFF       mov      bword ptr [rbp-0x260], rcx
       4533C0               xor      r8d, r8d
       4489855CFEFFFF       mov      dword ptr [rbp-0x1A4], r8d
       488B8DA0FDFFFF       mov      rcx, bword ptr [rbp-0x260]
       448B855CFEFFFF       mov      r8d, dword ptr [rbp-0x1A4]
       E929E5FFFF           jmp      G_M000_IG26
 
G_M000_IG281:                ;; offset=0x1D35
       33F6                 xor      rsi, rsi
       33D2                 xor      edx, edx
       48898DA0FDFFFF       mov      bword ptr [rbp-0x260], rcx
       4489855CFEFFFF       mov      dword ptr [rbp-0x1A4], r8d
       E935E5FFFF           jmp      G_M000_IG27
 
G_M000_IG282:                ;; offset=0x1D4C
       33FF                 xor      rdi, rdi
       33C0                 xor      eax, eax
       E949E5FFFF           jmp      G_M000_IG28
 
G_M000_IG283:                ;; offset=0x1D55
       4883F804             cmp      rax, 4
       7333                 jae      SHORT G_M000_IG289
 
G_M000_IG284:                ;; offset=0x1D5B
       33D2                 xor      edx, edx
       488BF0               mov      rsi, rax
       4883E602             and      rsi, 2
       740A                 je       SHORT G_M000_IG286
 
G_M000_IG285:                ;; offset=0x1D66
       0FB711               movzx    rdx, word  ptr [rcx]
       440FB707             movzx    r8, word  ptr [rdi]
       412BD0               sub      edx, r8d
 
G_M000_IG286:                ;; offset=0x1D70
       A801                 test     al, 1
       740D                 je       SHORT G_M000_IG288
 
G_M000_IG287:                ;; offset=0x1D74
       0FB60431             movzx    rax, byte  ptr [rcx+rsi]
       400FB63C37           movzx    rdi, byte  ptr [rdi+rsi]
       2BC7                 sub      eax, edi
       0BD0                 or       edx, eax
 
G_M000_IG288:                ;; offset=0x1D81
       85D2                 test     edx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E951FBFFFF           jmp      G_M000_IG220
 
G_M000_IG289:                ;; offset=0x1D8E
       4883C0FC             add      rax, -4
       8B11                 mov      edx, dword ptr [rcx]
       2B17                 sub      edx, dword ptr [rdi]
       8B0C01               mov      ecx, dword ptr [rcx+rax]
       2B0C07               sub      ecx, dword ptr [rdi+rax]
       0BCA                 or       ecx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E936FBFFFF           jmp      G_M000_IG220
 
G_M000_IG290:                ;; offset=0x1DA9
       62F17C4810040A       vmovups  zmm0, zmmword ptr [rdx+rcx]
       62F37D483E0C0E04     vpcmpfalseub k1, zmm0, zmmword ptr [rsi+rcx]
       C4E1F898C9           kortestq k1, k1
       0F85A0FBFFFF         jne      G_M000_IG229
       E931E5FFFF           jmp      G_M000_IG33
 
G_M000_IG291:                ;; offset=0x1DC8
       C5FC10043A           vmovups  ymm0, ymmword ptr [rdx+rdi]
       62F37D283E0C3E04     vpcmpfalseub k1, ymm0, ymmword ptr [rsi+rdi]
       C4E1F998C9           kortestd k1, k1
       0F8583FBFFFF         jne      G_M000_IG229
       E95AFBFFFF           jmp      G_M000_IG227
 
G_M000_IG292:                ;; offset=0x1DE5
       C5F810040A           vmovups  xmm0, xmmword ptr [rdx+rcx]
       62F37D083E0C0E04     vpcmpfalseub k1, xmm0, xmmword ptr [rsi+rcx]
       C5F898C9             kortestw k1, k1
       0F8567FBFFFF         jne      G_M000_IG229
       E915FBFFFF           jmp      G_M000_IG224
 
G_M000_IG293:                ;; offset=0x1E01
       33FF                 xor      edi, edi
       E91FE5FFFF           jmp      G_M000_IG36
 
G_M000_IG294:                ;; offset=0x1E08
       48BF28CD286A947C0000 mov      rdi, 0x7C946A28CD28
       E809DFF57C           call     CORINFO_HELP_NEWSFAST
       488BD8               mov      rbx, rax
       BF59E60000           mov      edi, 0xE659
       48BE087F346A947C0000 mov      rsi, 0x7C946A347F08
       FF15B97961FF         call     [CORINFO_HELP_STRCNS]
       4C8BE8               mov      r13, rax
       BF1FE90000           mov      edi, 0xE91F
       48BE087F346A947C0000 mov      rsi, 0x7C946A347F08
       FF15A17961FF         call     [CORINFO_HELP_STRCNS]
       488BF0               mov      rsi, rax
       498BFD               mov      rdi, r13
       FF15F5119BFE         call     [System.String:Concat(System.String,System.String):System.String]
       488BF0               mov      rsi, rax
       488BFB               mov      rdi, rbx
       FF15E97961FF         call     [System.ArgumentException:.ctor(System.String):this]
       488BFB               mov      rdi, rbx
       E8590BE17C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG295:                ;; offset=0x1E68
       488BFB               mov      rdi, rbx
       488B03               mov      rax, qword ptr [rbx]
       488B4078             mov      rax, qword ptr [rax+0x78]
       FF5010               call     [rax+0x10]Lokad.Onnx.Tensor`1[float]:ToDenseTensor():Lokad.Onnx.DenseTensor`1[float]:this
       488BF8               mov      rdi, rax
       498BF5               mov      rsi, r13
       FF15E703CCFF         call     [Lokad.Onnx.Tensor`1[float]:CountedCopy[float](Lokad.Onnx.DenseTensor`1[float],Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE0               mov      r12, rax
       E9B6E4FFFF           jmp      G_M000_IG37
 
G_M000_IG296:                ;; offset=0x1E89
       498BF6               mov      rsi, r14
       E8EF9CEAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BE0               mov      r12, rax
       E9D0E4FFFF           jmp      G_M000_IG40
 
G_M000_IG297:                ;; offset=0x1E99
       33C9                 xor      rcx, rcx
       48898D98FDFFFF       mov      bword ptr [rbp-0x268], rcx
       4533C0               xor      r8d, r8d
       44898558FEFFFF       mov      dword ptr [rbp-0x1A8], r8d
       488B8D98FDFFFF       mov      rcx, bword ptr [rbp-0x268]
       448B8558FEFFFF       mov      r8d, dword ptr [rbp-0x1A8]
       E9D5E4FFFF           jmp      G_M000_IG41
 
G_M000_IG298:                ;; offset=0x1EBF
       33F6                 xor      rsi, rsi
       33D2                 xor      edx, edx
       48898D98FDFFFF       mov      bword ptr [rbp-0x268], rcx
       44898558FEFFFF       mov      dword ptr [rbp-0x1A8], r8d
       E9E1E4FFFF           jmp      G_M000_IG42
 
G_M000_IG299:                ;; offset=0x1ED6
       33FF                 xor      rdi, rdi
       33C0                 xor      eax, eax
       E9F5E4FFFF           jmp      G_M000_IG43
 
G_M000_IG300:                ;; offset=0x1EDF
       4883F804             cmp      rax, 4
       7333                 jae      SHORT G_M000_IG306
 
G_M000_IG301:                ;; offset=0x1EE5
       33D2                 xor      edx, edx
       488BF0               mov      rsi, rax
       4883E602             and      rsi, 2
       740A                 je       SHORT G_M000_IG303
 
G_M000_IG302:                ;; offset=0x1EF0
       0FB711               movzx    rdx, word  ptr [rcx]
       440FB707             movzx    r8, word  ptr [rdi]
       412BD0               sub      edx, r8d
 
G_M000_IG303:                ;; offset=0x1EFA
       A801                 test     al, 1
       740D                 je       SHORT G_M000_IG305
 
G_M000_IG304:                ;; offset=0x1EFE
       0FB60431             movzx    rax, byte  ptr [rcx+rsi]
       400FB63C37           movzx    rdi, byte  ptr [rdi+rsi]
       2BC7                 sub      eax, edi
       0BD0                 or       edx, eax
 
G_M000_IG305:                ;; offset=0x1F0B
       85D2                 test     edx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E979FAFFFF           jmp      G_M000_IG233
 
G_M000_IG306:                ;; offset=0x1F18
       4883C0FC             add      rax, -4
       8B11                 mov      edx, dword ptr [rcx]
       2B17                 sub      edx, dword ptr [rdi]
       8B0C01               mov      ecx, dword ptr [rcx+rax]
       2B0C07               sub      ecx, dword ptr [rdi+rax]
       0BCA                 or       ecx, edx
       0F94C0               sete     al
       0FB6C0               movzx    rax, al
       E95EFAFFFF           jmp      G_M000_IG233
 
G_M000_IG307:                ;; offset=0x1F33
       62F17C4810040A       vmovups  zmm0, zmmword ptr [rdx+rcx]
       62F37D483E0C0E04     vpcmpfalseub k1, zmm0, zmmword ptr [rsi+rcx]
       C4E1F898C9           kortestq k1, k1
       0F85C8FAFFFF         jne      G_M000_IG242
       E9DDE4FFFF           jmp      G_M000_IG48
 
G_M000_IG308:                ;; offset=0x1F52
       C5FC10043A           vmovups  ymm0, ymmword ptr [rdx+rdi]
       62F37D283E0C3E04     vpcmpfalseub k1, ymm0, ymmword ptr [rsi+rdi]
       C4E1F998C9           kortestd k1, k1
       0F85ABFAFFFF         jne      G_M000_IG242
       E982FAFFFF           jmp      G_M000_IG240
 
G_M000_IG309:                ;; offset=0x1F6F
       C5F810040A           vmovups  xmm0, xmmword ptr [rdx+rcx]
       62F37D083E0C0E04     vpcmpfalseub k1, xmm0, xmmword ptr [rsi+rcx]
       C5F898C9             kortestw k1, k1
       0F858FFAFFFF         jne      G_M000_IG242
       E93DFAFFFF           jmp      G_M000_IG237
 
G_M000_IG310:                ;; offset=0x1F8B
       33FF                 xor      edi, edi
       E9CBE4FFFF           jmp      G_M000_IG51
 
G_M000_IG311:                ;; offset=0x1F92
       48BF28CD286A947C0000 mov      rdi, 0x7C946A28CD28
       E87FDDF57C           call     CORINFO_HELP_NEWSFAST
       4C8BF0               mov      r14, rax
       BF3AE80000           mov      edi, 0xE83A
       48BE087F346A947C0000 mov      rsi, 0x7C946A347F08
       FF152F7861FF         call     [CORINFO_HELP_STRCNS]
       4C8BE8               mov      r13, rax
       BF1FE90000           mov      edi, 0xE91F
       48BE087F346A947C0000 mov      rsi, 0x7C946A347F08
       FF15177861FF         call     [CORINFO_HELP_STRCNS]
       488BF0               mov      rsi, rax
       498BFD               mov      rdi, r13
       FF156B109BFE         call     [System.String:Concat(System.String,System.String):System.String]
       488BF0               mov      rsi, rax
       498BFE               mov      rdi, r14
       FF155F7861FF         call     [System.ArgumentException:.ctor(System.String):this]
       498BFE               mov      rdi, r14
       E8CF09E17C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG312:                ;; offset=0x1FF2
       498BFE               mov      rdi, r14
       498B06               mov      rax, qword ptr [r14]
       488B4078             mov      rax, qword ptr [rax+0x78]
       FF5010               call     [rax+0x10]Lokad.Onnx.Tensor`1[float]:ToDenseTensor():Lokad.Onnx.DenseTensor`1[float]:this
       488BF8               mov      rdi, rax
       498BF5               mov      rsi, r13
       FF155D02CCFF         call     [Lokad.Onnx.Tensor`1[float]:CountedCopy[float](Lokad.Onnx.DenseTensor`1[float],Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       4C8BE0               mov      r12, rax
       E962E4FFFF           jmp      G_M000_IG52
 
G_M000_IG313:                ;; offset=0x2013
       33FF                 xor      rdi, rdi
       33F6                 xor      esi, esi
       E970E4FFFF           jmp      G_M000_IG53
 
G_M000_IG314:                ;; offset=0x201C
       FF15968F14FF         call     [System.ThrowHelper:ThrowArgumentOutOfRangeException()]
       CC                   int3     
 
G_M000_IG315:                ;; offset=0x2023
       49BD80A84061947C0000 mov      r13, 0x7C946140A880
       E9BDE4FFFF           jmp      G_M000_IG54
 
G_M000_IG316:                ;; offset=0x2032
       33C0                 xor      rax, rax
       33FF                 xor      edi, edi
       E9D2E4FFFF           jmp      G_M000_IG55
 
G_M000_IG317:                ;; offset=0x203B
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E9F8E4FFFF           jmp      G_M000_IG56
 
G_M000_IG318:                ;; offset=0x2044
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E916E5FFFF           jmp      G_M000_IG57
 
G_M000_IG319:                ;; offset=0x204D
       33D2                 xor      rdx, rdx
       48899590FDFFFF       mov      bword ptr [rbp-0x270], rdx
       4533C0               xor      r8d, r8d
       44898550FEFFFF       mov      dword ptr [rbp-0x1B0], r8d
       488B9590FDFFFF       mov      rdx, bword ptr [rbp-0x270]
       448B8550FEFFFF       mov      r8d, dword ptr [rbp-0x1B0]
       E926E5FFFF           jmp      G_M000_IG58
 
G_M000_IG320:                ;; offset=0x2073
       48899590FDFFFF       mov      bword ptr [rbp-0x270], rdx
       44898550FEFFFF       mov      dword ptr [rbp-0x1B0], r8d
       498BF7               mov      rsi, r15
       48BF301A5C6B947C0000 mov      rdi, 0x7C946B5C1A30
       E8ED9AEAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BC8               mov      r9, rax
       488B9590FDFFFF       mov      rdx, bword ptr [rbp-0x270]
       448B8550FEFFFF       mov      r8d, dword ptr [rbp-0x1B0]
       E909E5FFFF           jmp      G_M000_IG59
 
G_M000_IG321:                ;; offset=0x20A9
       4983795000           cmp      gword ptr [r9+0x50], 0
       0F8407E5FFFF         je       G_M000_IG60
       4D8B5150             mov      r10, gword ptr [r9+0x50]
       4C8995E8FDFFFF       mov      gword ptr [rbp-0x218], r10
       E902E5FFFF           jmp      G_M000_IG61
 
G_M000_IG322:                ;; offset=0x20C4
       E95FF9FFFF           jmp      G_M000_IG244
 
G_M000_IG323:                ;; offset=0x20C9
       E95AF9FFFF           jmp      G_M000_IG244
 
G_M000_IG324:                ;; offset=0x20CE
       E955F9FFFF           jmp      G_M000_IG244
 
G_M000_IG325:                ;; offset=0x20D3
       413B7208             cmp      esi, dword ptr [r10+0x08]
       0F83CE010000         jae      G_M000_IG346
       8BC6                 mov      eax, esi
       418B448210           mov      eax, dword ptr [r10+4*rax+0x10]
       E96DF9FFFF           jmp      G_M000_IG247
 
G_M000_IG326:                ;; offset=0x20E9
       33C9                 xor      rcx, rcx
       48898D80FDFFFF       mov      bword ptr [rbp-0x280], rcx
       33D2                 xor      edx, edx
       899540FEFFFF         mov      dword ptr [rbp-0x1C0], edx
       488B8D80FDFFFF       mov      rcx, bword ptr [rbp-0x280]
       8B9540FEFFFF         mov      edx, dword ptr [rbp-0x1C0]
       E975E5FFFF           jmp      G_M000_IG72
 
G_M000_IG327:                ;; offset=0x210C
       48898D80FDFFFF       mov      bword ptr [rbp-0x280], rcx
       899540FEFFFF         mov      dword ptr [rbp-0x1C0], edx
       488BF3               mov      rsi, rbx
       48BF301A5C6B947C0000 mov      rdi, 0x7C946B5C1A30
       E8559AEAFF           call     CORINFO_HELP_ISINSTANCEOFCLASS
       4C8BC0               mov      r8, rax
       488B8D80FDFFFF       mov      rcx, bword ptr [rbp-0x280]
       8B9540FEFFFF         mov      edx, dword ptr [rbp-0x1C0]
       E95AE5FFFF           jmp      G_M000_IG73
 
G_M000_IG328:                ;; offset=0x2140
       4983785000           cmp      gword ptr [r8+0x50], 0
       0F8458E5FFFF         je       G_M000_IG74
       4D8B4850             mov      r9, gword ptr [r8+0x50]
       4C898DE0FDFFFF       mov      gword ptr [rbp-0x220], r9
       E953E5FFFF           jmp      G_M000_IG75
 
G_M000_IG329:                ;; offset=0x215B
       E91EF9FFFF           jmp      G_M000_IG249
 
G_M000_IG330:                ;; offset=0x2160
       E919F9FFFF           jmp      G_M000_IG249
 
G_M000_IG331:                ;; offset=0x2165
       E914F9FFFF           jmp      G_M000_IG249
 
G_M000_IG332:                ;; offset=0x216A
       413B7108             cmp      esi, dword ptr [r9+0x08]
       0F8337010000         jae      G_M000_IG346
       8BCE                 mov      ecx, esi
       418B4C8910           mov      ecx, dword ptr [r9+4*rcx+0x10]
       E92CF9FFFF           jmp      G_M000_IG252
 
G_M000_IG333:                ;; offset=0x2180
       33C9                 xor      rcx, rcx
       33D2                 xor      edx, edx
       E9D8E5FFFF           jmp      G_M000_IG86
 
G_M000_IG334:                ;; offset=0x2189
       E943F9FFFF           jmp      G_M000_IG254
 
G_M000_IG335:                ;; offset=0x218E
       E93EF9FFFF           jmp      G_M000_IG254
 
G_M000_IG336:                ;; offset=0x2193
       E939F9FFFF           jmp      G_M000_IG254
 
G_M000_IG337:                ;; offset=0x2198
       3B4708               cmp      eax, dword ptr [rdi+0x08]
       0F830A010000         jae      G_M000_IG346
       8BF0                 mov      esi, eax
       8B4CB710             mov      ecx, dword ptr [rdi+4*rsi+0x10]
       E953F9FFFF           jmp      G_M000_IG257
 
G_M000_IG338:                ;; offset=0x21AC
       837DD402             cmp      dword ptr [rbp-0x2C], 2
       0F8C8BE6FFFF         jl       G_M000_IG100
       8B4870               mov      ecx, dword ptr [rax+0x70]
       448B45D4             mov      r8d, dword ptr [rbp-0x2C]
       413BC8               cmp      ecx, r8d
       410F4FC8             cmovg    ecx, r8d
       898D38FFFFFF         mov      dword ptr [rbp-0xC8], ecx
       E97CE6FFFF           jmp      G_M000_IG101
 
G_M000_IG339:                ;; offset=0x21CF
       BA56555555           mov      edx, 0x55555556
       8BC2                 mov      eax, edx
       41F7ED               imul     edx:eax, r13d
       8BC2                 mov      eax, edx
       C1E81F               shr      eax, 31
       03C2                 add      eax, edx
       8D0440               lea      eax, [rax+2*rax]
       442BE8               sub      r13d, eax
       0F85A0E7FFFF         jne      G_M000_IG112
       E986E6FFFF           jmp      G_M000_IG102
 
G_M000_IG340:                ;; offset=0x21F1
       33C0                 xor      eax, eax
       E9AAE6FFFF           jmp      G_M000_IG104
 
G_M000_IG341:                ;; offset=0x21F8
       33C9                 xor      rcx, rcx
       33C0                 xor      eax, eax
       E9BDE6FFFF           jmp      G_M000_IG105
 
G_M000_IG342:                ;; offset=0x2201
       33C9                 xor      rcx, rcx
       33FF                 xor      edi, edi
       E9CEE6FFFF           jmp      G_M000_IG106
 
G_M000_IG343:                ;; offset=0x220A
       48BA882D80818C7C0000 mov      rdx, 0x7C8C81802D88
       488B12               mov      rdx, gword ptr [rdx]
       4885D2               test     rdx, rdx
       7556                 jne      SHORT G_M000_IG344
       48BFE82F386B947C0000 mov      rdi, 0x7C946B382FE8
       E8F5DAF57C           call     CORINFO_HELP_NEWSFAST
       488BD0               mov      rdx, rax
       488995F8FDFFFF       mov      gword ptr [rbp-0x208], rdx
       48BE702B80818C7C0000 mov      rsi, 0x7C8C81802B70
       488B36               mov      rsi, gword ptr [rsi]
       488BFA               mov      rdi, rdx
       48BAC09E556B947C0000 mov      rdx, 0x7C946B559EC0
       FF15939F9AFE         call     [System.MulticastDelegate:CtorClosed(System.Object,nint):this]
       48BF882D80818C7C0000 mov      rdi, 0x7C8C81802D88
       488BB5F8FDFFFF       mov      rsi, gword ptr [rbp-0x208]
       E8B5319AFD           call     CORINFO_HELP_ASSIGN_REF
       488B95F8FDFFFF       mov      rdx, gword ptr [rbp-0x208]
 
G_M000_IG344:                ;; offset=0x2272
       488BBD10FEFFFF       mov      rdi, gword ptr [rbp-0x1F0]
       488BF2               mov      rsi, rdx
       FF15D677B8FF         call     [System.Linq.Enumerable:All[int](System.Collections.Generic.IEnumerable`1[int],System.Func`2[int,bool]):bool]
       85C0                 test     eax, eax
       0F8407E7FFFF         je       G_M000_IG113
       E995E6FFFF           jmp      G_M000_IG109
 
G_M000_IG345:                ;; offset=0x228F
       488B7DB8             mov      rdi, gword ptr [rbp-0x48]
       49BBB8262669947C0000 mov      r11, 0x7C94692626B8
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33C0                 xor      rax, rax
       488945B8             mov      gword ptr [rbp-0x48], rax
       E950F5FFFF           jmp      G_M000_IG203
 
G_M000_IG346:                ;; offset=0x22AB
       E8A0489AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG347:                ;; offset=0x22B1
       4883EC38             sub      rsp, 56
 
G_M000_IG348:                ;; offset=0x22B5
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG349:                ;; offset=0x22BE
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG350:                ;; offset=0x22C6
       4883EC38             sub      rsp, 56
 
G_M000_IG351:                ;; offset=0x22CA
       48BF682E80818C7C0000 mov      rdi, 0x7C8C81802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB5C0FDFFFF       mov      rsi, gword ptr [rbp-0x240]
       33D2                 xor      edx, edx
       FF15BABED5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       90                   nop      
 
G_M000_IG352:                ;; offset=0x22E7
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG353:                ;; offset=0x22EF
       4883EC38             sub      rsp, 56
 
G_M000_IG354:                ;; offset=0x22F3
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG355:                ;; offset=0x22FC
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG356:                ;; offset=0x2304
       4883EC38             sub      rsp, 56
 
G_M000_IG357:                ;; offset=0x2308
       48BF682E80818C7C0000 mov      rdi, 0x7C8C81802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB5C8FDFFFF       mov      rsi, gword ptr [rbp-0x238]
       33D2                 xor      edx, edx
       FF157CBED5FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       90                   nop      
 
G_M000_IG358:                ;; offset=0x2325
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG359:                ;; offset=0x232D
       4883EC38             sub      rsp, 56
 
G_M000_IG360:                ;; offset=0x2331
       33FF                 xor      rdi, rdi
       4889BD88FEFFFF       mov      gword ptr [rbp-0x178], rdi
 
G_M000_IG361:                ;; offset=0x233A
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG362:                ;; offset=0x2342
       4883EC38             sub      rsp, 56
 
G_M000_IG363:                ;; offset=0x2346
       48BF682E80818C7C0000 mov      rdi, 0x7C8C81802E68
       4C8B37               mov      r14, gword ptr [rdi]
       4883BDD0FDFFFF00     cmp      gword ptr [rbp-0x230], 0
       750C                 jne      SHORT G_M000_IG365
 
G_M000_IG364:                ;; offset=0x235D
       BF02000000           mov      edi, 2
       FF15A01FAEFF         call     [System.ThrowHelper:ThrowArgumentNullException(int)]
       CC                   int3     
 
G_M000_IG365:                ;; offset=0x2369
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       8B7F08               mov      edi, dword ptr [rdi+0x08]
       FFCF                 dec      edi
       83CF0F               or       edi, 15
       33DB                 xor      ebx, ebx
       F30FBDDF             lzcnt    ebx, edi
       83F31F               xor      ebx, 31
       83C3FD               add      ebx, -3
       48BFE8EC89E8947C0000 mov      rdi, 0x7C94E889ECE8
       48B820A81DE9947C0000 mov      rax, 0x7C94E91DA820
       FFD0                 call     rax
       833809               cmp      dword ptr [rax], 9
       7E0D                 jle      SHORT G_M000_IG366
       488B7808             mov      rdi, gword ptr [rax+0x08]
       488B4748             mov      rax, bword ptr [rdi+0x48]
       4885C0               test     rax, rax
       750A                 jne      SHORT G_M000_IG367
 
G_M000_IG366:                ;; offset=0x23AC
       BF09000000           mov      edi, 9
       E8DAB7F0FF           call     CORINFO_HELP_GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED
 
G_M000_IG367:                ;; offset=0x23B6
       488B4010             mov      rax, gword ptr [rax+0x10]
       4885C0               test     rax, rax
       7509                 jne      SHORT G_M000_IG369
 
G_M000_IG368:                ;; offset=0x23BF
       498BFE               mov      rdi, r14
       FF15981ECCFF         call     [System.Buffers.SharedArrayPool`1[float]:InitializeTlsBucketsAndTrimming():System.Buffers.SharedArrayPoolThreadLocalArray[]:this]
 
G_M000_IG369:                ;; offset=0x23C8
       4533FF               xor      r15d, r15d
       41BD01000000         mov      r13d, 1
       395808               cmp      dword ptr [rax+0x08], ebx
       0F863C020000         jbe      G_M000_IG387
 
G_M000_IG370:                ;; offset=0x23DA
       41BF01000000         mov      r15d, 1
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       BE10000000           mov      esi, 16
       C4E261F7F6           shlx     esi, esi, ebx
       397708               cmp      dword ptr [rdi+0x08], esi
       7448                 je       SHORT G_M000_IG372
 
G_M000_IG371:                ;; offset=0x23F6
       48BF28CD286A947C0000 mov      rdi, 0x7C946A28CD28
       E81BD9F57C           call     CORINFO_HELP_NEWSFAST
       4C8BE0               mov      r12, rax
       FF156A1ECCFF         call     [System.SR:get_ArgumentException_BufferNotFromPool():System.String]
       488BD8               mov      rbx, rax
       BF6D040000           mov      edi, 0x46D
       48BE00402569947C0000 mov      rsi, 0x7C9469254000
       FF15C27361FF         call     [CORINFO_HELP_STRCNS]
       488BD0               mov      rdx, rax
       488BF3               mov      rsi, rbx
       498BFC               mov      rdi, r12
       FF15D37461FF         call     [System.ArgumentException:.ctor(System.String,System.String):this]
       498BFC               mov      rdi, r12
       E88305E17C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG372:                ;; offset=0x243E
       3B5808               cmp      ebx, dword ptr [rax+0x08]
       0F83A2020000         jae      G_M000_IG391
       8BFB                 mov      edi, ebx
       48C1E704             shl      rdi, 4
       4C8D643810           lea      r12, bword ptr [rax+rdi+0x10]
       498B0424             mov      rax, gword ptr [r12]
       488985B8FDFFFF       mov      gword ptr [rbp-0x248], rax
       488BB5D0FDFFFF       mov      rsi, gword ptr [rbp-0x230]
       498BFC               mov      rdi, r12
       E8B42F9AFD           call     CORINFO_HELP_ASSIGN_REF
       33FF                 xor      edi, edi
       41897C2408           mov      dword ptr [r12+0x08], edi
       4C8BA5B8FDFFFF       mov      r12, gword ptr [rbp-0x248]
       4D85E4               test     r12, r12
       0F8493010000         je       G_M000_IG387
 
G_M000_IG373:                ;; offset=0x2483
       498B7E10             mov      rdi, gword ptr [r14+0x10]
       3B5F08               cmp      ebx, dword ptr [rdi+0x08]
       0F8359020000         jae      G_M000_IG391
       8BF3                 mov      esi, ebx
       488B44F710           mov      rax, gword ptr [rdi+8*rsi+0x10]
       4885C0               test     rax, rax
       750B                 jne      SHORT G_M000_IG374
       498BFE               mov      rdi, r14
       8BF3                 mov      esi, ebx
       FF15011ECCFF         call     [System.Buffers.SharedArrayPool`1[float]:CreatePerCorePartitions(int):System.Buffers.SharedArrayPoolPartitions:this]
 
G_M000_IG374:                ;; offset=0x24A7
       4C8B6808             mov      r13, gword ptr [rax+0x08]
       48BF50BA7E6B947C0000 mov      rdi, 0x7C946B7EBA50
       FF154D869AFE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       803D7AE29BFD00       cmp      byte  ptr [(reloc 0x7c946925b35c)], 0
       7412                 je       SHORT G_M000_IG375
       C5F877               vzeroupper 
       E874B899FE           call     Interop+Sys:SchedGetCpu():int
       8BD0                 mov      edx, eax
       899570FEFFFF         mov      dword ptr [rbp-0x190], edx
       EB4B                 jmp      SHORT G_M000_IG377
 
G_M000_IG375:                ;; offset=0x24D6
       BF0A000000           mov      edi, 10
       FF15EF16E1FF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B4010               mov      eax, dword ptr [rax+0x10]
       89856CFEFFFF         mov      dword ptr [rbp-0x194], eax
       BF0A000000           mov      edi, 10
       FF15DB16E1FF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B956CFEFFFF         mov      edx, dword ptr [rbp-0x194]
       8D4AFF               lea      ecx, [rdx-0x01]
       894810               mov      dword ptr [rax+0x10], ecx
       0FB7C2               movzx    rax, dx
       85C0                 test     eax, eax
       7510                 jne      SHORT G_M000_IG376
       FF15DA16E1FF         call     [System.Threading.ProcessorIdCache:RefreshCurrentProcessorId():int]
       8BD0                 mov      edx, eax
       899570FEFFFF         mov      dword ptr [rbp-0x190], edx
       EB09                 jmp      SHORT G_M000_IG377
 
G_M000_IG376:                ;; offset=0x2518
       C1FA10               sar      edx, 16
       899570FEFFFF         mov      dword ptr [rbp-0x190], edx
 
G_M000_IG377:                ;; offset=0x2521
       48BFF0B87E6B947C0000 mov      rdi, 0x7C946B7EB8F0
       FF15D7859AFE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       8B8570FEFFFF         mov      eax, dword ptr [rbp-0x190]
       33D2                 xor      edx, edx
       F735F1E19BFD         div      edx:eax, dword ptr [(reloc 0x7c946925b350)]
       8BC2                 mov      eax, edx
       33C9                 xor      ecx, ecx
       E9B1000000           jmp      G_M000_IG384
 
G_M000_IG378:                ;; offset=0x2548
       413B4508             cmp      eax, dword ptr [r13+0x08]
       0F8397010000         jae      G_M000_IG391
       898578FEFFFF         mov      dword ptr [rbp-0x188], eax
       8BF8                 mov      edi, eax
       498B54FD10           mov      rdx, gword ptr [r13+8*rdi+0x10]
       488995B0FDFFFF       mov      gword ptr [rbp-0x250], rdx
       3812                 cmp      byte  ptr [rdx], dl
       33F6                 xor      esi, esi
       89B568FEFFFF         mov      dword ptr [rbp-0x198], esi
       488BFA               mov      rdi, rdx
       FF15AF9BF7FF         call     [System.Threading.Monitor:Enter(System.Object)]
       488B85B0FDFFFF       mov      rax, gword ptr [rbp-0x250]
       488B7808             mov      rdi, gword ptr [rax+0x08]
       8B4810               mov      ecx, dword ptr [rax+0x10]
       898D64FEFFFF         mov      dword ptr [rbp-0x19C], ecx
       394F08               cmp      dword ptr [rdi+0x08], ecx
       7635                 jbe      SHORT G_M000_IG380
       85C9                 test     ecx, ecx
       7545                 jne      SHORT G_M000_IG381
       33F6                 xor      esi, esi
       897014               mov      dword ptr [rax+0x14], esi
 
G_M000_IG379:                ;; offset=0x259B
       4863F1               movsxd   rsi, ecx
       488D7CF710           lea      rdi, bword ptr [rdi+8*rsi+0x10]
       498BF4               mov      rsi, r12
       E8752E9AFD           call     CORINFO_HELP_ASSIGN_REF
       8BBD64FEFFFF         mov      edi, dword ptr [rbp-0x19C]
       FFC7                 inc      edi
       488B85B0FDFFFF       mov      rax, gword ptr [rbp-0x250]
       897810               mov      dword ptr [rax+0x10], edi
       C78568FEFFFF01000000 mov      dword ptr [rbp-0x198], 1
 
G_M000_IG380:                ;; offset=0x25C7
       488BF8               mov      rdi, rax
       FF1518969AFE         call     [System.Threading.Monitor:Exit(System.Object)]
       83BD68FEFFFF00       cmp      dword ptr [rbp-0x198], 0
       7404                 je       SHORT G_M000_IG382
       EB30                 jmp      SHORT G_M000_IG385
 
G_M000_IG381:                ;; offset=0x25DB
       EBBE                 jmp      SHORT G_M000_IG379
 
G_M000_IG382:                ;; offset=0x25DD
       8B8578FEFFFF         mov      eax, dword ptr [rbp-0x188]
       FFC0                 inc      eax
       8BF8                 mov      edi, eax
       41397D08             cmp      dword ptr [r13+0x08], edi
       7502                 jne      SHORT G_M000_IG383
       33FF                 xor      edi, edi
 
G_M000_IG383:                ;; offset=0x25EF
       8B8D74FEFFFF         mov      ecx, dword ptr [rbp-0x18C]
       FFC1                 inc      ecx
       8BC7                 mov      eax, edi
 
G_M000_IG384:                ;; offset=0x25F9
       898D74FEFFFF         mov      dword ptr [rbp-0x18C], ecx
       41394D08             cmp      dword ptr [r13+0x08], ecx
       0F8F3FFFFFFF         jg       G_M000_IG378
       EB08                 jmp      SHORT G_M000_IG386
 
G_M000_IG385:                ;; offset=0x260B
       41BD01000000         mov      r13d, 1
       EB03                 jmp      SHORT G_M000_IG387
 
G_M000_IG386:                ;; offset=0x2613
       4533ED               xor      r13d, r13d
 
G_M000_IG387:                ;; offset=0x2616
       48BFF80180818C7C0000 mov      rdi, 0x7C8C818001F8
       4C8B27               mov      r12, gword ptr [rdi]
       4180BC249D00000000   cmp      byte  ptr [r12+0x9D], 0
       0F84BD000000         je       G_M000_IG392
 
G_M000_IG388:                ;; offset=0x2632
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       837F0800             cmp      dword ptr [rdi+0x08], 0
       0F84AC000000         je       G_M000_IG392
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       FF15609A73FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       898584FEFFFF         mov      dword ptr [rbp-0x17C], eax
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       8B4F08               mov      ecx, dword ptr [rdi+0x08]
       898D80FEFFFF         mov      dword ptr [rbp-0x180], ecx
       498BFE               mov      rdi, r14
       FF15419A73FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       448BC0               mov      r8d, eax
       498BFC               mov      rdi, r12
       8B9584FEFFFF         mov      edx, dword ptr [rbp-0x17C]
       8B8D80FEFFFF         mov      ecx, dword ptr [rbp-0x180]
       BE03000000           mov      esi, 3
       FF15242CE1FF         call     [System.Diagnostics.Tracing.EventSource:WriteEvent(int,int,int,int):this]
       4585FD               test     r15d, r13d
       755E                 jne      SHORT G_M000_IG392
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       FF15129A73FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       448BE8               mov      r13d, eax
       488BBDD0FDFFFF       mov      rdi, gword ptr [rbp-0x230]
       8B4708               mov      eax, dword ptr [rdi+0x08]
       89857CFEFFFF         mov      dword ptr [rbp-0x184], eax
       498BFE               mov      rdi, r14
       FF15F69973FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       4585FF               test     r15d, r15d
       750E                 jne      SHORT G_M000_IG389
       41B8FFFFFFFF         mov      r8d, -1
       41B901000000         mov      r9d, 1
       EB06                 jmp      SHORT G_M000_IG390
 
G_M000_IG389:                ;; offset=0x26CF
       448BC3               mov      r8d, ebx
       4533C9               xor      r9d, r9d
 
G_M000_IG390:                ;; offset=0x26D5
       498BFC               mov      rdi, r12
       418BF5               mov      esi, r13d
       8B957CFEFFFF         mov      edx, dword ptr [rbp-0x184]
       FF15091CCCFF         call     [System.Buffers.ArrayPoolEventSource:BufferDropped(int,int,int,int,int):this]
       EB06                 jmp      SHORT G_M000_IG392
 
G_M000_IG391:                ;; offset=0x26E9
       E862449AFE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG392:                ;; offset=0x26EF
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG393:                ;; offset=0x26F7
       4883EC38             sub      rsp, 56
 
G_M000_IG394:                ;; offset=0x26FB
       48837D9800           cmp      qword ptr [rbp-0x68], 0
       740A                 je       SHORT G_M000_IG396
 
G_M000_IG395:                ;; offset=0x2702
       488D7D98             lea      rdi, [rbp-0x68]
       FF15C40EA1FF         call     [System.Runtime.InteropServices.GCHandle:Free():this]
 
G_M000_IG396:                ;; offset=0x270C
       48837D8800           cmp      gword ptr [rbp-0x78], 0
       7417                 je       SHORT G_M000_IG398
 
G_M000_IG397:                ;; offset=0x2713
       488B7D88             mov      rdi, gword ptr [rbp-0x78]
       49BBA8262669947C0000 mov      r11, 0x7C94692626A8
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33FF                 xor      rdi, rdi
       48897D88             mov      gword ptr [rbp-0x78], rdi
 
G_M000_IG398:                ;; offset=0x272A
       33FF                 xor      edi, edi
       48897D90             mov      qword ptr [rbp-0x70], rdi
 
G_M000_IG399:                ;; offset=0x2730
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG400:                ;; offset=0x2738
       4883EC38             sub      rsp, 56
 
G_M000_IG401:                ;; offset=0x273C
       48837DB000           cmp      qword ptr [rbp-0x50], 0
       740A                 je       SHORT G_M000_IG403
 
G_M000_IG402:                ;; offset=0x2743
       488D7DB0             lea      rdi, [rbp-0x50]
       FF15830EA1FF         call     [System.Runtime.InteropServices.GCHandle:Free():this]
 
G_M000_IG403:                ;; offset=0x274D
       48837DA000           cmp      gword ptr [rbp-0x60], 0
       7417                 je       SHORT G_M000_IG405
 
G_M000_IG404:                ;; offset=0x2754
       488B7DA0             mov      rdi, gword ptr [rbp-0x60]
       49BBB0262669947C0000 mov      r11, 0x7C94692626B0
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG405:                ;; offset=0x276B
       33FF                 xor      edi, edi
       48897DA8             mov      qword ptr [rbp-0x58], rdi
 
G_M000_IG406:                ;; offset=0x2771
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG407:                ;; offset=0x2779
       4883EC38             sub      rsp, 56
 
G_M000_IG408:                ;; offset=0x277D
       48837DC800           cmp      qword ptr [rbp-0x38], 0
       740A                 je       SHORT G_M000_IG410
 
G_M000_IG409:                ;; offset=0x2784
       488D7DC8             lea      rdi, [rbp-0x38]
       FF15420EA1FF         call     [System.Runtime.InteropServices.GCHandle:Free():this]
 
G_M000_IG410:                ;; offset=0x278E
       48837DB800           cmp      gword ptr [rbp-0x48], 0
       7417                 je       SHORT G_M000_IG412
 
G_M000_IG411:                ;; offset=0x2795
       488B7DB8             mov      rdi, gword ptr [rbp-0x48]
       49BBB8262669947C0000 mov      r11, 0x7C94692626B8
       41FF13               call     [r11]System.Buffers.IPinnable:Unpin():this
       33FF                 xor      rdi, rdi
       48897DB8             mov      gword ptr [rbp-0x48], rdi
 
G_M000_IG412:                ;; offset=0x27AC
       33FF                 xor      edi, edi
       48897DC0             mov      qword ptr [rbp-0x40], rdi
 
G_M000_IG413:                ;; offset=0x27B2
       C5F877               vzeroupper 
       4883C438             add      rsp, 56
       C3                   ret      
 
; Total bytes of code 10170
