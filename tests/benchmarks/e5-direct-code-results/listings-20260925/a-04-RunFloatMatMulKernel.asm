; Assembly listing for method Lokad.Onnx.Tensor`1[float]:RunFloatMatMulKernel(int,int,int,ptr,ptr,ptr,Lokad.Onnx.TensorExecutionOptions) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 164928
; 13 inlinees with PGO data; 46 single block inlinees; 4 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       55                   push     rbp
       4157                 push     r15
       4156                 push     r14
       4155                 push     r13
       4154                 push     r12
       53                   push     rbx
       4881EC08010000       sub      rsp, 264
       C5F877               vzeroupper 
       488DAC2430010000     lea      rbp, [rsp+0x130]
       33C0                 xor      eax, eax
       8945AC               mov      dword ptr [rbp-0x54], eax
       488945A0             mov      qword ptr [rbp-0x60], rax
       897DD4               mov      dword ptr [rbp-0x2C], edi
       8975D0               mov      dword ptr [rbp-0x30], esi
       8955CC               mov      dword ptr [rbp-0x34], edx
       48894DC0             mov      qword ptr [rbp-0x40], rcx
       4C8945B8             mov      qword ptr [rbp-0x48], r8
       4C894DB0             mov      qword ptr [rbp-0x50], r9
 
G_M000_IG02:                ;; offset=0x003A
       0FB65D3C             movzx    rbx, byte  ptr [rbp+0x3C]
       85DB                 test     ebx, ebx
       0F84FD020000         je       G_M000_IG30
 
G_M000_IG03:                ;; offset=0x0046
       440FB67D3D           movzx    r15, byte  ptr [rbp+0x3D]
       4585FF               test     r15d, r15d
       7409                 je       SHORT G_M000_IG04
       83FF01               cmp      edi, 1
       0F8406030000         je       G_M000_IG33
 
G_M000_IG04:                ;; offset=0x0059
       4585FF               test     r15d, r15d
       0F8458090000         je       G_M000_IG81
       83FF02               cmp      edi, 2
       0F8C4F090000         jl       G_M000_IG81
       8BC7                 mov      eax, edi
       83E001               and      eax, 1
       8BD7                 mov      edx, edi
       2BD0                 sub      edx, eax
       8955AC               mov      dword ptr [rbp-0x54], edx
       B8ABAAAAAA           mov      eax, 0xAAAAAAAB
       8BD7                 mov      edx, edi
       480FAFC2             imul     rax, rdx
       48C1E821             shr      rax, 33
       8D0440               lea      eax, [rax+2*rax]
       448BFF               mov      r15d, edi
       442BF8               sub      r15d, eax
       0F84E7020000         je       G_M000_IG34
 
G_M000_IG05:                ;; offset=0x0095
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       837DAC40             cmp      dword ptr [rbp-0x54], 64
       0F8DDF030000         jge      G_M000_IG41
 
G_M000_IG06:                ;; offset=0x00A5
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       4585FF               test     r15d, r15d
       740D                 je       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x00AD
       40F6C701             test     dil, 1
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       0F8523080000         jne      G_M000_IG72
 
G_M000_IG08:                ;; offset=0x00BA
       4863C6               movsxd   rax, esi
       486355CC             movsxd   rdx, dword ptr [rbp-0x34]
       480FAFC2             imul     rax, rdx
       483D00000100         cmp      rax, 0x10000
       0F8F0C080000         jg       G_M000_IG72
 
G_M000_IG09:                ;; offset=0x00D1
       8BDE                 mov      ebx, esi
       0FAF5DCC             imul     ebx, dword ptr [rbp-0x34]
       4C8B7510             mov      r14, gword ptr [rbp+0x10]
       48B8682E80295F7A0000 mov      rax, 0x7A5F29802E68
       4C8B28               mov      r13, gword ptr [rax]
       4D8BE5               mov      r12, r13
       8BC3                 mov      eax, ebx
       48BAF80180295F7A0000 mov      rdx, 0x7A5F298001F8
       488B12               mov      rdx, gword ptr [rdx]
       48899528FFFFFF       mov      gword ptr [rbp-0xD8], rdx
       89459C               mov      dword ptr [rbp-0x64], eax
       448D50FF             lea      r10d, [rax-0x01]
       4183CA0F             or       r10d, 15
       F3450FBDD2           lzcnt    r10d, r10d
       4183F21F             xor      r10d, 31
       4183C2FD             add      r10d, -3
       44895598             mov      dword ptr [rbp-0x68], r10d
       48BFE8ECA98E677A0000 mov      rdi, 0x7A678EA9ECE8
       49BB20B83C8F677A0000 mov      r11, 0x7A678F3CB820
       41FFD3               call     r11
       833809               cmp      dword ptr [rax], 9
       0F8E44040000         jle      G_M000_IG48
       488B7808             mov      rdi, gword ptr [rax+0x08]
       488B4748             mov      rax, bword ptr [rdi+0x48]
       4885C0               test     rax, rax
       0F8433040000         je       G_M000_IG48
 
G_M000_IG10:                ;; offset=0x014E
       488B7810             mov      rdi, gword ptr [rax+0x10]
       4885FF               test     rdi, rdi
       0F848A040000         je       G_M000_IG51
       8B4708               mov      eax, dword ptr [rdi+0x08]
       8B4D98               mov      ecx, dword ptr [rbp-0x68]
       3BC1                 cmp      eax, ecx
       0F867F040000         jbe      G_M000_IG52
       8BC1                 mov      eax, ecx
       48C1E004             shl      rax, 4
       488B540710           mov      rdx, gword ptr [rdi+rax+0x10]
       4885D2               test     rdx, rdx
       0F8463040000         je       G_M000_IG50
       33F6                 xor      rsi, rsi
       4889740710           mov      gword ptr [rdi+rax+0x10], rsi
       4C8BA528FFFFFF       mov      r12, gword ptr [rbp-0xD8]
       4180BC249D00000000   cmp      byte  ptr [r12+0x9D], 0
       0F85F6030000         jne      G_M000_IG49
 
G_M000_IG11:                ;; offset=0x019A
       4C8BEA               mov      r13, rdx
 
G_M000_IG12:                ;; offset=0x019D
       4D85F6               test     r14, r14
       7422                 je       SHORT G_M000_IG13
       48BEA0697111677A0000 mov      rsi, 0x7A67117169A0
       493936               cmp      qword ptr [r14], rsi
       0F8504070000         jne      G_M000_IG70
       4983C608             add      r14, 8
       4863F3               movsxd   rsi, ebx
       48C1E602             shl      rsi, 2
       F0                   lock     
       490136               add      qword ptr [r14], rsi
 
G_M000_IG13:                ;; offset=0x01C4
       4C89AD30FFFFFF       mov      gword ptr [rbp-0xD0], r13
 
G_M000_IG14:                ;; offset=0x01CB
       4C896DA0             mov      gword ptr [rbp-0x60], r13
       4D85ED               test     r13, r13
       0F843F010000         je       G_M000_IG26
 
G_M000_IG15:                ;; offset=0x01D8
       41837D0800           cmp      dword ptr [r13+0x08], 0
       0F8434010000         je       G_M000_IG26
       4983C510             add      r13, 16
       4D8BC5               mov      r8, r13
 
G_M000_IG16:                ;; offset=0x01EA
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       8BFA                 mov      edi, edx
       C1FF1F               sar      edi, 31
       83E71F               and      edi, 31
       03FA                 add      edi, edx
       83E7E0               and      edi, -32
       8BF2                 mov      esi, edx
       2BF7                 sub      esi, edi
       8BFA                 mov      edi, edx
       2BFE                 sub      edi, esi
       33F6                 xor      esi, esi
       85FF                 test     edi, edi
       0F8E7C000000         jle      G_M000_IG20
 
G_M000_IG17:                ;; offset=0x020C
       8BCE                 mov      ecx, esi
       C1F91F               sar      ecx, 31
       83E11F               and      ecx, 31
       03CE                 add      ecx, esi
       C1F905               sar      ecx, 5
       0FAF4DD0             imul     ecx, dword ptr [rbp-0x30]
       C1E105               shl      ecx, 5
       4863C9               movsxd   rcx, ecx
       498D0C88             lea      rcx, [r8+4*rcx]
       4533C9               xor      r9d, r9d
       837DD000             cmp      dword ptr [rbp-0x30], 0
       7E4A                 jle      SHORT G_M000_IG19
                            align    [0 bytes for IG18]
 
G_M000_IG18:                ;; offset=0x0230
       418BC1               mov      eax, r9d
       C1E005               shl      eax, 5
       4898                 cdqe     
       488D0481             lea      rax, [rcx+4*rax]
       458BD1               mov      r10d, r9d
       440FAFD2             imul     r10d, edx
       4D63D2               movsxd   r10, r10d
       49C1E202             shl      r10, 2
       4C0355B8             add      r10, qword ptr [rbp-0x48]
       4C63DE               movsxd   r11, esi
       4F8D149A             lea      r10, [r10+4*r11]
       62D17E486F02         vmovdqu32 zmm0, zmmword ptr [r10]
       62D17E486F4A01       vmovdqu32 zmm1, zmmword ptr [r10+0x40]
       62F17E487F00         vmovdqu32 zmmword ptr [rax], zmm0
       62F17E487F4801       vmovdqu32 zmmword ptr [rax+0x40], zmm1
       41FFC1               inc      r9d
       8B45D0               mov      eax, dword ptr [rbp-0x30]
       443BC8               cmp      r9d, eax
       7CB6                 jl       SHORT G_M000_IG18
 
G_M000_IG19:                ;; offset=0x027A
       8B45D0               mov      eax, dword ptr [rbp-0x30]
       83C620               add      esi, 32
       3BF7                 cmp      esi, edi
       7C88                 jl       SHORT G_M000_IG17
 
G_M000_IG20:                ;; offset=0x0284
       8BF2                 mov      esi, edx
       2BF7                 sub      esi, edi
       8BCF                 mov      ecx, edi
       C1F91F               sar      ecx, 31
       83E11F               and      ecx, 31
       03CF                 add      ecx, edi
       C1F905               sar      ecx, 5
       8B45D0               mov      eax, dword ptr [rbp-0x30]
       0FAFC8               imul     ecx, eax
       C1E105               shl      ecx, 5
       4863C9               movsxd   rcx, ecx
       498D0C88             lea      rcx, [r8+4*rcx]
       4533C9               xor      r9d, r9d
       85C0                 test     eax, eax
       7E4D                 jle      SHORT G_M000_IG25
 
G_M000_IG21:                ;; offset=0x02AC
       4863FF               movsxd   rdi, edi
       48C1E702             shl      rdi, 2
 
G_M000_IG22:                ;; offset=0x02B3
       458BD1               mov      r10d, r9d
       440FAFD2             imul     r10d, edx
       4D63D2               movsxd   r10, r10d
       49C1E202             shl      r10, 2
       4C0355B8             add      r10, qword ptr [rbp-0x48]
       4C03D7               add      r10, rdi
       458BD9               mov      r11d, r9d
       440FAFDE             imul     r11d, esi
       4D63DB               movsxd   r11, r11d
       4E8D1C99             lea      r11, [rcx+4*r11]
       33DB                 xor      ebx, ebx
       85F6                 test     esi, esi
       7E15                 jle      SHORT G_M000_IG24
                            align    [0 bytes for IG23]
 
G_M000_IG23:                ;; offset=0x02DC
       4C63F3               movsxd   r14, ebx
       C4817A1004B2         vmovss   xmm0, dword ptr [r10+4*r14]
       C4817A1104B3         vmovss   dword ptr [r11+4*r14], xmm0
       FFC3                 inc      ebx
       3BDE                 cmp      ebx, esi
       7CEB                 jl       SHORT G_M000_IG23
 
G_M000_IG24:                ;; offset=0x02F1
       41FFC1               inc      r9d
       443BC8               cmp      r9d, eax
       7CBA                 jl       SHORT G_M000_IG22
 
G_M000_IG25:                ;; offset=0x02F9
       4585FF               test     r15d, r15d
       741D                 je       SHORT G_M000_IG27
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8BF0                 mov      esi, eax
       488B4DC0             mov      rcx, qword ptr [rbp-0x40]
       4C8B4DB0             mov      r9, qword ptr [rbp-0x50]
       FF151F50DAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       EB1C                 jmp      SHORT G_M000_IG28
 
G_M000_IG26:                ;; offset=0x0313
       4533C0               xor      r8d, r8d
       E9CFFEFFFF           jmp      G_M000_IG16
 
G_M000_IG27:                ;; offset=0x031B
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8BF0                 mov      esi, eax
       488B4DC0             mov      rcx, qword ptr [rbp-0x40]
       4C8B4DB0             mov      r9, qword ptr [rbp-0x50]
       FF15EA4FDAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG28:                ;; offset=0x032F
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG29:                ;; offset=0x0335
       E82B070000           call     G_M000_IG99
       E9C5050000           jmp      G_M000_IG75
 
G_M000_IG30:                ;; offset=0x033F
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       FF15205ADAFF         call     [Lokad.Onnx.MathOps:mm(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG31:                ;; offset=0x0348
       90                   nop      
 
G_M000_IG32:                ;; offset=0x0349
       4881C408010000       add      rsp, 264
       5B                   pop      rbx
       415C                 pop      r12
       415D                 pop      r13
       415E                 pop      r14
       415F                 pop      r15
       5D                   pop      rbp
       C3                   ret      
 
G_M000_IG33:                ;; offset=0x035B
       817DCC00200000       cmp      dword ptr [rbp-0x34], 0x2000
       0F8CF1FCFFFF         jl       G_M000_IG04
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       BF01000000           mov      edi, 1
       FF157A59DAFF         call     [Lokad.Onnx.MathOps:mm_m1_kblocked(int,int,int,ptr,ptr,ptr)]
       EBD0                 jmp      SHORT G_M000_IG31
 
G_M000_IG34:                ;; offset=0x0378
       83FF40               cmp      edi, 64
       0F8C14FDFFFF         jl       G_M000_IG05
       4863C6               movsxd   rax, esi
       486355CC             movsxd   rdx, dword ptr [rbp-0x34]
       480FAFC2             imul     rax, rdx
       483D00000004         cmp      rax, 0x4000000
       0F8FFDFCFFFF         jg       G_M000_IG05
       885D3C               mov      byte  ptr [rbp+0x3C], bl
       488D3C24             lea      rdi, [rsp]
       488D7510             lea      rsi, [rbp+0x10]
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
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       8BFE                 mov      edi, esi
       0FAF7DCC             imul     edi, dword ptr [rbp-0x34]
       FF15BD58DAFF         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       48898540FFFFFF       mov      gword ptr [rbp-0xC0], rax
 
G_M000_IG35:                ;; offset=0x040A
       488B8540FFFFFF       mov      rax, gword ptr [rbp-0xC0]
       488945A0             mov      gword ptr [rbp-0x60], rax
       4885C0               test     rax, rax
       7406                 je       SHORT G_M000_IG36
       83780800             cmp      dword ptr [rax+0x08], 0
       7504                 jne      SHORT G_M000_IG37
 
G_M000_IG36:                ;; offset=0x0420
       33DB                 xor      ebx, ebx
       EB04                 jmp      SHORT G_M000_IG38
 
G_M000_IG37:                ;; offset=0x0424
       488D5810             lea      rbx, bword ptr [rax+0x10]
 
G_M000_IG38:                ;; offset=0x0428
       8B7DD0               mov      edi, dword ptr [rbp-0x30]
       8B75CC               mov      esi, dword ptr [rbp-0x34]
       488B55B8             mov      rdx, qword ptr [rbp-0x48]
       488BCB               mov      rcx, rbx
       FF15E5D2C6FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       488B4DC0             mov      rcx, qword ptr [rbp-0x40]
       4C8BC3               mov      r8, rbx
       4C8B4DB0             mov      r9, qword ptr [rbp-0x50]
       FF15C34EDAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG39:                ;; offset=0x0456
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG40:                ;; offset=0x045C
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB540FFFFFF       mov      rsi, gword ptr [rbp-0xC0]
       33D2                 xor      edx, edx
       FF15A8FBE3FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       E9A4040000           jmp      G_M000_IG76
 
G_M000_IG41:                ;; offset=0x0480
       4863C6               movsxd   rax, esi
       486355CC             movsxd   rdx, dword ptr [rbp-0x34]
       480FAFC2             imul     rax, rdx
       483D00000004         cmp      rax, 0x4000000
       0F8F0EFCFFFF         jg       G_M000_IG06
       885D3C               mov      byte  ptr [rbp+0x3C], bl
       488D3C24             lea      rdi, [rsp]
       488D7510             lea      rsi, [rbp+0x10]
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
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       8BFE                 mov      edi, esi
       0FAF7DCC             imul     edi, dword ptr [rbp-0x34]
       FF15BE57DAFF         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       48898538FFFFFF       mov      gword ptr [rbp-0xC8], rax
 
G_M000_IG42:                ;; offset=0x0509
       488B8538FFFFFF       mov      rax, gword ptr [rbp-0xC8]
       488945A0             mov      gword ptr [rbp-0x60], rax
       4885C0               test     rax, rax
       7406                 je       SHORT G_M000_IG43
       83780800             cmp      dword ptr [rax+0x08], 0
       7505                 jne      SHORT G_M000_IG44
 
G_M000_IG43:                ;; offset=0x051F
       4533FF               xor      r15d, r15d
       EB04                 jmp      SHORT G_M000_IG45
 
G_M000_IG44:                ;; offset=0x0524
       4C8D7810             lea      r15, bword ptr [rax+0x10]
 
G_M000_IG45:                ;; offset=0x0528
       8B7DD0               mov      edi, dword ptr [rbp-0x30]
       8B75CC               mov      esi, dword ptr [rbp-0x34]
       488B55B8             mov      rdx, qword ptr [rbp-0x48]
       498BCF               mov      rcx, r15
       FF15E5D1C6FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8B7DAC               mov      edi, dword ptr [rbp-0x54]
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       488B4DC0             mov      rcx, qword ptr [rbp-0x40]
       4D8BC7               mov      r8, r15
       4C8B4DB0             mov      r9, qword ptr [rbp-0x50]
       FF15DB4DDAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG46:                ;; offset=0x0556
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG47:                ;; offset=0x055C
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB538FFFFFF       mov      rsi, gword ptr [rbp-0xC8]
       33D2                 xor      edx, edx
       FF15A8FAE3FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       E987030000           jmp      G_M000_IG75
 
G_M000_IG48:                ;; offset=0x057D
       BF09000000           mov      edi, 9
       E80904FFFF           call     CORINFO_HELP_GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED
       E9C2FBFFFF           jmp      G_M000_IG10
 
G_M000_IG49:                ;; offset=0x058C
       48899520FFFFFF       mov      gword ptr [rbp-0xE0], rdx
       488BFA               mov      rdi, rdx
       FF1594D981FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       894580               mov      dword ptr [rbp-0x80], eax
       488B8D20FFFFFF       mov      rcx, gword ptr [rbp-0xE0]
       8B5108               mov      edx, dword ptr [rcx+0x08]
       89957CFFFFFF         mov      dword ptr [rbp-0x84], edx
       498BFD               mov      rdi, r13
       FF1578D981FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       8B7580               mov      esi, dword ptr [rbp-0x80]
       8B957CFFFFFF         mov      edx, dword ptr [rbp-0x84]
       498BFC               mov      rdi, r12
       448B4598             mov      r8d, dword ptr [rbp-0x68]
       FF15F059DAFF         call     [System.Buffers.ArrayPoolEventSource:BufferRented(int,int,int,int):this]
       488B9520FFFFFF       mov      rdx, gword ptr [rbp-0xE0]
       E9BEFBFFFF           jmp      G_M000_IG11
 
G_M000_IG50:                ;; offset=0x05DC
       8B4D98               mov      ecx, dword ptr [rbp-0x68]
       EB03                 jmp      SHORT G_M000_IG52
 
G_M000_IG51:                ;; offset=0x05E1
       8B4D98               mov      ecx, dword ptr [rbp-0x68]
 
G_M000_IG52:                ;; offset=0x05E4
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       394F08               cmp      dword ptr [rdi+0x08], ecx
       0F86CC010000         jbe      G_M000_IG65
       8BC1                 mov      eax, ecx
       488B7CC710           mov      rdi, gword ptr [rdi+8*rax+0x10]
       4885FF               test     rdi, rdi
       0F84AA010000         je       G_M000_IG64
       4C8B6F08             mov      r13, gword ptr [rdi+0x08]
       48BF10AE9E11677A0000 mov      rdi, 0x7A67119EAE10
       FF1572C3A8FE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       803D9F1FAAFD00       cmp      byte  ptr [(reloc 0x7a670f45b35c)], 0
       740F                 je       SHORT G_M000_IG53
       E89CF5A7FE           call     Interop+Sys:SchedGetCpu():int
       8BD0                 mov      edx, eax
       899570FFFFFF         mov      dword ptr [rbp-0x90], edx
       EB4B                 jmp      SHORT G_M000_IG55
 
G_M000_IG53:                ;; offset=0x062E
       BF0A000000           mov      edi, 10
       FF15FF53EFFF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B4010               mov      eax, dword ptr [rax+0x10]
       89856CFFFFFF         mov      dword ptr [rbp-0x94], eax
       BF0A000000           mov      edi, 10
       FF15EB53EFFF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B956CFFFFFF         mov      edx, dword ptr [rbp-0x94]
       8D4AFF               lea      ecx, [rdx-0x01]
       894810               mov      dword ptr [rax+0x10], ecx
       0FB7C2               movzx    rax, dx
       85C0                 test     eax, eax
       7510                 jne      SHORT G_M000_IG54
       FF15EA53EFFF         call     [System.Threading.ProcessorIdCache:RefreshCurrentProcessorId():int]
       8BD0                 mov      edx, eax
       899570FFFFFF         mov      dword ptr [rbp-0x90], edx
       EB09                 jmp      SHORT G_M000_IG55
 
G_M000_IG54:                ;; offset=0x0670
       C1FA10               sar      edx, 16
       899570FFFFFF         mov      dword ptr [rbp-0x90], edx
 
G_M000_IG55:                ;; offset=0x0679
       48BFB0AC9E11677A0000 mov      rdi, 0x7A67119EACB0
       FF15FFC2A8FE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       8B8570FFFFFF         mov      eax, dword ptr [rbp-0x90]
       33D2                 xor      edx, edx
       F735191FAAFD         div      edx:eax, dword ptr [(reloc 0x7a670f45b350)]
       8BC2                 mov      eax, edx
       33C9                 xor      ecx, ecx
       E999000000           jmp      G_M000_IG59
 
G_M000_IG56:                ;; offset=0x06A0
       413B4508             cmp      eax, dword ptr [r13+0x08]
       0F832D030000         jae      G_M000_IG83
       898578FFFFFF         mov      dword ptr [rbp-0x88], eax
       8BF8                 mov      edi, eax
       498B54FD10           mov      rdx, gword ptr [r13+8*rdi+0x10]
       48899510FFFFFF       mov      gword ptr [rbp-0xF0], rdx
       3812                 cmp      byte  ptr [rdx], dl
       33F6                 xor      rsi, rsi
       4889B518FFFFFF       mov      gword ptr [rbp-0xE8], rsi
       488BFA               mov      rdi, rdx
       FF15EED80400         call     [System.Threading.Monitor:Enter(System.Object)]
       488B9510FFFFFF       mov      rdx, gword ptr [rbp-0xF0]
       488B7A08             mov      rdi, gword ptr [rdx+0x08]
       8B4210               mov      eax, dword ptr [rdx+0x10]
       FFC8                 dec      eax
       394708               cmp      dword ptr [rdi+0x08], eax
       761B                 jbe      SHORT G_M000_IG57
       8BC8                 mov      ecx, eax
       488B74CF10           mov      rsi, gword ptr [rdi+8*rcx+0x10]
       4889B518FFFFFF       mov      gword ptr [rbp-0xE8], rsi
       8BF0                 mov      esi, eax
       4533C0               xor      r8, r8
       4C8944F710           mov      gword ptr [rdi+8*rsi+0x10], r8
       894210               mov      dword ptr [rdx+0x10], eax
 
G_M000_IG57:                ;; offset=0x0702
       488BFA               mov      rdi, rdx
       FF155DD3A8FE         call     [System.Threading.Monitor:Exit(System.Object)]
       488B8D18FFFFFF       mov      rcx, gword ptr [rbp-0xE8]
       4885C9               test     rcx, rcx
       7534                 jne      SHORT G_M000_IG60
       8B8578FFFFFF         mov      eax, dword ptr [rbp-0x88]
       FFC0                 inc      eax
       8BC8                 mov      ecx, eax
       41394D08             cmp      dword ptr [r13+0x08], ecx
       7506                 jne      SHORT G_M000_IG58
       33C9                 xor      ecx, ecx
       33FF                 xor      edi, edi
       8BCF                 mov      ecx, edi
 
G_M000_IG58:                ;; offset=0x072D
       8BBD74FFFFFF         mov      edi, dword ptr [rbp-0x8C]
       FFC7                 inc      edi
       8BC1                 mov      eax, ecx
       8BCF                 mov      ecx, edi
 
G_M000_IG59:                ;; offset=0x0739
       898D74FFFFFF         mov      dword ptr [rbp-0x8C], ecx
       41394D08             cmp      dword ptr [r13+0x08], ecx
       0F8F57FFFFFF         jg       G_M000_IG56
       EB02                 jmp      SHORT G_M000_IG61
 
G_M000_IG60:                ;; offset=0x074B
       EB02                 jmp      SHORT G_M000_IG62
 
G_M000_IG61:                ;; offset=0x074D
       33C9                 xor      rcx, rcx
 
G_M000_IG62:                ;; offset=0x074F
       4C8BE9               mov      r13, rcx
       4D85ED               test     r13, r13
       7455                 je       SHORT G_M000_IG64
       488B8528FFFFFF       mov      rax, gword ptr [rbp-0xD8]
       80B89D00000000       cmp      byte  ptr [rax+0x9D], 0
       743D                 je       SHORT G_M000_IG63
       498BFD               mov      rdi, r13
       FF15C0D781FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       894588               mov      dword ptr [rbp-0x78], eax
       418B4D08             mov      ecx, dword ptr [r13+0x08]
       894D84               mov      dword ptr [rbp-0x7C], ecx
       498BFC               mov      rdi, r12
       FF15ADD781FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       8B7588               mov      esi, dword ptr [rbp-0x78]
       8B5584               mov      edx, dword ptr [rbp-0x7C]
       488BBD28FFFFFF       mov      rdi, gword ptr [rbp-0xD8]
       448B4598             mov      r8d, dword ptr [rbp-0x68]
       FF152458DAFF         call     [System.Buffers.ArrayPoolEventSource:BufferRented(int,int,int,int):this]
       498BD5               mov      rdx, r13
       E9F6F9FFFF           jmp      G_M000_IG11
 
G_M000_IG63:                ;; offset=0x07A4
       498BD5               mov      rdx, r13
       E9EEF9FFFF           jmp      G_M000_IG11
 
G_M000_IG64:                ;; offset=0x07AC
       BF10000000           mov      edi, 16
       8B4D98               mov      ecx, dword ptr [rbp-0x68]
       C4E271F7C7           shlx     eax, edi, ecx
       448BE8               mov      r13d, eax
       EB2B                 jmp      SHORT G_M000_IG67
 
G_M000_IG65:                ;; offset=0x07BE
       448B6D9C             mov      r13d, dword ptr [rbp-0x64]
       4585ED               test     r13d, r13d
       750F                 jne      SHORT G_M000_IG66
       49BDE0AE800C677A0000 mov      r13, 0x7A670C80AEE0
       E9C7F9FFFF           jmp      G_M000_IG12
 
G_M000_IG66:                ;; offset=0x07D6
       418BFD               mov      edi, r13d
       48BEF810800C677A0000 mov      rsi, 0x7A670C8010F8
       FF159F2F23FF         call     [System.ArgumentOutOfRangeException:ThrowIfNegative[int](int,System.String)]
 
G_M000_IG67:                ;; offset=0x07E9
       4181FD00020000       cmp      r13d, 512
       7D17                 jge      SHORT G_M000_IG68
       4963F5               movsxd   rsi, r13d
       48BF788F5410677A0000 mov      rdi, 0x7A6710548F78
       E8AC14047D           call     CORINFO_HELP_NEWARR_1_VC
       4C8BE8               mov      r13, rax
       EB0E                 jmp      SHORT G_M000_IG69
 
G_M000_IG68:                ;; offset=0x0809
       418BFD               mov      edi, r13d
       33F6                 xor      esi, esi
       FF150C58DAFF         call     [System.GC:<AllocateUninitializedArray>g__AllocateNewArrayWorker|77_0[float](int,bool):float[]]
       4C8BE8               mov      r13, rax
 
G_M000_IG69:                ;; offset=0x0817
       488B8528FFFFFF       mov      rax, gword ptr [rbp-0xD8]
       80B89D00000000       cmp      byte  ptr [rax+0x9D], 0
       0F84A6000000         je       G_M000_IG71
       45386D00             cmp      byte  ptr [r13], r13b
       498BFD               mov      rdi, r13
       FF15F8D681FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       894594               mov      dword ptr [rbp-0x6C], eax
       418B4D08             mov      ecx, dword ptr [r13+0x08]
       894D90               mov      dword ptr [rbp-0x70], ecx
       498BFC               mov      rdi, r12
       FF15E5D681FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       8B7594               mov      esi, dword ptr [rbp-0x6C]
       8B5590               mov      edx, dword ptr [rbp-0x70]
       488BBD28FFFFFF       mov      rdi, gword ptr [rbp-0xD8]
       41B8FFFFFFFF         mov      r8d, -1
       FF155A57DAFF         call     [System.Buffers.ArrayPoolEventSource:BufferRented(int,int,int,int):this]
       418B4508             mov      eax, dword ptr [r13+0x08]
       89458C               mov      dword ptr [rbp-0x74], eax
       498BFC               mov      rdi, r12
       FF15BAD681FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       41B901000000         mov      r9d, 1
       BE02000000           mov      esi, 2
       448B6598             mov      r12d, dword ptr [rbp-0x68]
       44396708             cmp      dword ptr [rdi+0x08], r12d
       440F4FCE             cmovg    r9d, esi
       488BBD28FFFFFF       mov      rdi, gword ptr [rbp-0xD8]
       8B7594               mov      esi, dword ptr [rbp-0x6C]
       8B558C               mov      edx, dword ptr [rbp-0x74]
       41B8FFFFFFFF         mov      r8d, -1
       FF155B57DAFF         call     [System.Buffers.ArrayPoolEventSource:BufferAllocated(int,int,int,int,int):this]
       498BD5               mov      rdx, r13
       E9E5F8FFFF           jmp      G_M000_IG11
 
G_M000_IG70:                ;; offset=0x08B5
       4863F3               movsxd   rsi, ebx
       48C1E602             shl      rsi, 2
       498BFE               mov      rdi, r14
       49BB0825460F677A0000 mov      r11, 0x7A670F462508
       41FF13               call     [r11]Lokad.Onnx.IScratchAccountant:AddScratchBytes(long):this
       E9F3F8FFFF           jmp      G_M000_IG13
 
G_M000_IG71:                ;; offset=0x08D1
       498BD5               mov      rdx, r13
       E9C1F8FFFF           jmp      G_M000_IG11
 
G_M000_IG72:                ;; offset=0x08D9
       81FE000A0000         cmp      esi, 0xA00
       7D17                 jge      SHORT G_M000_IG74
 
G_M000_IG73:                ;; offset=0x08E1
       817DCC000A0000       cmp      dword ptr [rbp-0x34], 0xA00
       7D0E                 jge      SHORT G_M000_IG74
       8B7DAC               mov      edi, dword ptr [rbp-0x54]
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       FF151254DAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4tiled(int,int,int,ptr,ptr,ptr)]
       EB0C                 jmp      SHORT G_M000_IG75
 
G_M000_IG74:                ;; offset=0x08F8
       8B7DAC               mov      edi, dword ptr [rbp-0x54]
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       FF151C54DAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG75:                ;; offset=0x0904
       41B9ABAAAAAA         mov      r9d, 0xAAAAAAAB
       8B4DD4               mov      ecx, dword ptr [rbp-0x2C]
       4C0FAFC9             imul     r9, rcx
       49C1E921             shr      r9, 33
       478D0C49             lea      r9d, [r9+2*r9]
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       448BFF               mov      r15d, edi
       452BF9               sub      r15d, r9d
       753F                 jne      SHORT G_M000_IG78
 
G_M000_IG76:                ;; offset=0x0924
       4C634DD0             movsxd   r9, dword ptr [rbp-0x30]
       48634DCC             movsxd   rcx, dword ptr [rbp-0x34]
       4C0FAFC9             imul     r9, rcx
       4981F900000004       cmp      r9, 0x4000000
       7F2A                 jg       SHORT G_M000_IG78
       83FF40               cmp      edi, 64
       7D1D                 jge      SHORT G_M000_IG77
       4C634DD0             movsxd   r9, dword ptr [rbp-0x30]
       48634DCC             movsxd   rcx, dword ptr [rbp-0x34]
       4C0FAFC9             imul     r9, rcx
       4981F900000100       cmp      r9, 0x10000
       410F9EC1             setle    r9b
       450FB6C9             movzx    r9, r9b
       EB0B                 jmp      SHORT G_M000_IG79
 
G_M000_IG77:                ;; offset=0x095B
       41B901000000         mov      r9d, 1
       EB03                 jmp      SHORT G_M000_IG79
 
G_M000_IG78:                ;; offset=0x0963
       4533C9               xor      r9d, r9d
 
G_M000_IG79:                ;; offset=0x0966
       397DAC               cmp      dword ptr [rbp-0x54], edi
       0F84D9F9FFFF         je       G_M000_IG31
 
G_M000_IG80:                ;; offset=0x096F
       4585C9               test     r9d, r9d
       0F85D0F9FFFF         jne      G_M000_IG31
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       448BCA               mov      r9d, edx
       440FAF4DAC           imul     r9d, dword ptr [rbp-0x54]
       4D63C9               movsxd   r9, r9d
       488B45B0             mov      rax, qword ptr [rbp-0x50]
       4E8D0C88             lea      r9, [rax+4*r9]
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       8BCE                 mov      ecx, esi
       0FAF4DAC             imul     ecx, dword ptr [rbp-0x54]
       4863C9               movsxd   rcx, ecx
       4C8B55C0             mov      r10, qword ptr [rbp-0x40]
       498D0C8A             lea      rcx, [r10+4*rcx]
       4C8B45B8             mov      r8, qword ptr [rbp-0x48]
       BF01000000           mov      edi, 1
       FF158753DAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E992F9FFFF           jmp      G_M000_IG31
 
G_M000_IG81:                ;; offset=0x09B6
       4585FF               test     r15d, r15d
       740E                 je       SHORT G_M000_IG82
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       FF157453DAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       E97FF9FFFF           jmp      G_M000_IG31
 
G_M000_IG82:                ;; offset=0x09C9
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       FF157E53DAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized(int,int,int,ptr,ptr,ptr)]
       E971F9FFFF           jmp      G_M000_IG31
 
G_M000_IG83:                ;; offset=0x09D7
       E8F47FA8FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG84:                ;; offset=0x09DD
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG85:                ;; offset=0x09E4
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG86:                ;; offset=0x09EA
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG87:                ;; offset=0x09EF
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG88:                ;; offset=0x09F6
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB540FFFFFF       mov      rsi, gword ptr [rbp-0xC0]
       33D2                 xor      edx, edx
       FF150EF6E3FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       90                   nop      
 
G_M000_IG89:                ;; offset=0x0A13
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG90:                ;; offset=0x0A18
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG91:                ;; offset=0x0A1F
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG92:                ;; offset=0x0A25
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG93:                ;; offset=0x0A2A
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG94:                ;; offset=0x0A31
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB538FFFFFF       mov      rsi, gword ptr [rbp-0xC8]
       33D2                 xor      edx, edx
       FF15D3F5E3FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       90                   nop      
 
G_M000_IG95:                ;; offset=0x0A4E
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG96:                ;; offset=0x0A53
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG97:                ;; offset=0x0A5A
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG98:                ;; offset=0x0A60
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG99:                ;; offset=0x0A65
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG100:                ;; offset=0x0A6C
       48BF682E80295F7A0000 mov      rdi, 0x7A5F29802E68
       4C8B3F               mov      r15, gword ptr [rdi]
       4883BD30FFFFFF00     cmp      gword ptr [rbp-0xD0], 0
       750C                 jne      SHORT G_M000_IG102
 
G_M000_IG101:                ;; offset=0x0A83
       BF02000000           mov      edi, 2
       FF15FA56BCFF         call     [System.ThrowHelper:ThrowArgumentNullException(int)]
       CC                   int3     
 
G_M000_IG102:                ;; offset=0x0A8F
       488BBD30FFFFFF       mov      rdi, gword ptr [rbp-0xD0]
       8B4708               mov      eax, dword ptr [rdi+0x08]
       FFC8                 dec      eax
       83C80F               or       eax, 15
       33DB                 xor      ebx, ebx
       F30FBDD8             lzcnt    ebx, eax
       83F31F               xor      ebx, 31
       83C3FD               add      ebx, -3
       48BFE8ECA98E677A0000 mov      rdi, 0x7A678EA9ECE8
       48B820B83C8F677A0000 mov      rax, 0x7A678F3CB820
       FFD0                 call     rax
       833809               cmp      dword ptr [rax], 9
       7E0D                 jle      SHORT G_M000_IG103
       488B7808             mov      rdi, gword ptr [rax+0x08]
       488B4748             mov      rax, bword ptr [rdi+0x48]
       4885C0               test     rax, rax
       750A                 jne      SHORT G_M000_IG104
 
G_M000_IG103:                ;; offset=0x0AD2
       BF09000000           mov      edi, 9
       E8B4FEFEFF           call     CORINFO_HELP_GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED
 
G_M000_IG104:                ;; offset=0x0ADC
       488B4010             mov      rax, gword ptr [rax+0x10]
       4885C0               test     rax, rax
       7509                 jne      SHORT G_M000_IG106
 
G_M000_IG105:                ;; offset=0x0AE5
       498BFF               mov      rdi, r15
       FF15F255DAFF         call     [System.Buffers.SharedArrayPool`1[float]:InitializeTlsBucketsAndTrimming():System.Buffers.SharedArrayPoolThreadLocalArray[]:this]
 
G_M000_IG106:                ;; offset=0x0AEE
       4533F6               xor      r14d, r14d
       41BD01000000         mov      r13d, 1
       395808               cmp      dword ptr [rax+0x08], ebx
       0F862B020000         jbe      G_M000_IG124
 
G_M000_IG107:                ;; offset=0x0B00
       41BE01000000         mov      r14d, 1
       BF10000000           mov      edi, 16
       C4E261F7FF           shlx     edi, edi, ebx
       488B8D30FFFFFF       mov      rcx, gword ptr [rbp-0xD0]
       397908               cmp      dword ptr [rcx+0x08], edi
       7448                 je       SHORT G_M000_IG109
 
G_M000_IG108:                ;; offset=0x0B1C
       48BF28CD4810677A0000 mov      rdi, 0x7A671048CD28
       E87510047D           call     CORINFO_HELP_NEWSFAST
       4C8BF8               mov      r15, rax
       FF15C455DAFF         call     [System.SR:get_ArgumentException_BufferNotFromPool():System.String]
       4C8BF0               mov      r14, rax
       BF6D040000           mov      edi, 0x46D
       48BE0040450F677A0000 mov      rsi, 0x7A670F454000
       FF151CAB6FFF         call     [CORINFO_HELP_STRCNS]
       488BD0               mov      rdx, rax
       498BF6               mov      rsi, r14
       498BFF               mov      rdi, r15
       FF152DAC6FFF         call     [System.ArgumentException:.ctor(System.String,System.String):this]
       498BFF               mov      rdi, r15
       E8DD3CEF7C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG109:                ;; offset=0x0B64
       8BFB                 mov      edi, ebx
       48C1E704             shl      rdi, 4
       4C8D643810           lea      r12, bword ptr [rax+rdi+0x10]
       498B0424             mov      rax, gword ptr [r12]
       48898508FFFFFF       mov      gword ptr [rbp-0xF8], rax
       498BFC               mov      rdi, r12
       488BF1               mov      rsi, rcx
       E81B67A8FD           call     CORINFO_HELP_ASSIGN_REF
       33FF                 xor      edi, edi
       41897C2408           mov      dword ptr [r12+0x08], edi
       4C8BA508FFFFFF       mov      r12, gword ptr [rbp-0xF8]
       4D85E4               test     r12, r12
       0F848F010000         je       G_M000_IG124
 
G_M000_IG110:                ;; offset=0x0B9C
       498B7F10             mov      rdi, gword ptr [r15+0x10]
       3B5F08               cmp      ebx, dword ptr [rdi+0x08]
       0F8351020000         jae      G_M000_IG128
       8BF3                 mov      esi, ebx
       488B44F710           mov      rax, gword ptr [rdi+8*rsi+0x10]
       4885C0               test     rax, rax
       750B                 jne      SHORT G_M000_IG111
       498BFF               mov      rdi, r15
       8BF3                 mov      esi, ebx
       FF156855DAFF         call     [System.Buffers.SharedArrayPool`1[float]:CreatePerCorePartitions(int):System.Buffers.SharedArrayPoolPartitions:this]
 
G_M000_IG111:                ;; offset=0x0BC0
       4C8B6808             mov      r13, gword ptr [rax+0x08]
       48BF10AE9E11677A0000 mov      rdi, 0x7A67119EAE10
       FF15B4BDA8FE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       803DE119AAFD00       cmp      byte  ptr [(reloc 0x7a670f45b35c)], 0
       740F                 je       SHORT G_M000_IG112
       E8DEEFA7FE           call     Interop+Sys:SchedGetCpu():int
       8BD0                 mov      edx, eax
       899554FFFFFF         mov      dword ptr [rbp-0xAC], edx
       EB4B                 jmp      SHORT G_M000_IG114
 
G_M000_IG112:                ;; offset=0x0BEC
       BF0A000000           mov      edi, 10
       FF15414EEFFF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B4010               mov      eax, dword ptr [rax+0x10]
       898550FFFFFF         mov      dword ptr [rbp-0xB0], eax
       BF0A000000           mov      edi, 10
       FF152D4EEFFF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B9550FFFFFF         mov      edx, dword ptr [rbp-0xB0]
       8D4AFF               lea      ecx, [rdx-0x01]
       894810               mov      dword ptr [rax+0x10], ecx
       0FB7C2               movzx    rax, dx
       85C0                 test     eax, eax
       7510                 jne      SHORT G_M000_IG113
       FF152C4EEFFF         call     [System.Threading.ProcessorIdCache:RefreshCurrentProcessorId():int]
       8BD0                 mov      edx, eax
       899554FFFFFF         mov      dword ptr [rbp-0xAC], edx
       EB09                 jmp      SHORT G_M000_IG114
 
G_M000_IG113:                ;; offset=0x0C2E
       C1FA10               sar      edx, 16
       899554FFFFFF         mov      dword ptr [rbp-0xAC], edx
 
G_M000_IG114:                ;; offset=0x0C37
       48BFB0AC9E11677A0000 mov      rdi, 0x7A67119EACB0
       FF1541BDA8FE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       8B8554FFFFFF         mov      eax, dword ptr [rbp-0xAC]
       33D2                 xor      edx, edx
       F7355B19AAFD         div      edx:eax, dword ptr [(reloc 0x7a670f45b350)]
       8BC2                 mov      eax, edx
       33C9                 xor      ecx, ecx
       E9B0000000           jmp      G_M000_IG121
 
G_M000_IG115:                ;; offset=0x0C5E
       413B4508             cmp      eax, dword ptr [r13+0x08]
       0F8392010000         jae      G_M000_IG128
       89855CFFFFFF         mov      dword ptr [rbp-0xA4], eax
       8BF8                 mov      edi, eax
       498B54FD10           mov      rdx, gword ptr [r13+8*rdi+0x10]
       48899500FFFFFF       mov      gword ptr [rbp-0x100], rdx
       3812                 cmp      byte  ptr [rdx], dl
       33F6                 xor      esi, esi
       89B54CFFFFFF         mov      dword ptr [rbp-0xB4], esi
       488BFA               mov      rdi, rdx
       FF1531D30400         call     [System.Threading.Monitor:Enter(System.Object)]
       488B8500FFFFFF       mov      rax, gword ptr [rbp-0x100]
       488B7808             mov      rdi, gword ptr [rax+0x08]
       8B4810               mov      ecx, dword ptr [rax+0x10]
       898D48FFFFFF         mov      dword ptr [rbp-0xB8], ecx
       394F08               cmp      dword ptr [rdi+0x08], ecx
       7634                 jbe      SHORT G_M000_IG117
       85C9                 test     ecx, ecx
       7544                 jne      SHORT G_M000_IG118
       33F6                 xor      esi, esi
       897014               mov      dword ptr [rax+0x14], esi
 
G_M000_IG116:                ;; offset=0x0CB1
       8BF1                 mov      esi, ecx
       488D7CF710           lea      rdi, bword ptr [rdi+8*rsi+0x10]
       498BF4               mov      rsi, r12
       E8E065A8FD           call     CORINFO_HELP_ASSIGN_REF
       8BBD48FFFFFF         mov      edi, dword ptr [rbp-0xB8]
       FFC7                 inc      edi
       488B8500FFFFFF       mov      rax, gword ptr [rbp-0x100]
       897810               mov      dword ptr [rax+0x10], edi
       C7854CFFFFFF01000000 mov      dword ptr [rbp-0xB4], 1
 
G_M000_IG117:                ;; offset=0x0CDC
       488BF8               mov      rdi, rax
       FF1583CDA8FE         call     [System.Threading.Monitor:Exit(System.Object)]
       83BD4CFFFFFF00       cmp      dword ptr [rbp-0xB4], 0
       7404                 je       SHORT G_M000_IG119
       EB30                 jmp      SHORT G_M000_IG122
 
G_M000_IG118:                ;; offset=0x0CF0
       EBBF                 jmp      SHORT G_M000_IG116
 
G_M000_IG119:                ;; offset=0x0CF2
       8B855CFFFFFF         mov      eax, dword ptr [rbp-0xA4]
       FFC0                 inc      eax
       8BF8                 mov      edi, eax
       41397D08             cmp      dword ptr [r13+0x08], edi
       7502                 jne      SHORT G_M000_IG120
       33FF                 xor      edi, edi
 
G_M000_IG120:                ;; offset=0x0D04
       8B8D58FFFFFF         mov      ecx, dword ptr [rbp-0xA8]
       FFC1                 inc      ecx
       8BC7                 mov      eax, edi
 
G_M000_IG121:                ;; offset=0x0D0E
       898D58FFFFFF         mov      dword ptr [rbp-0xA8], ecx
       41394D08             cmp      dword ptr [r13+0x08], ecx
       0F8F40FFFFFF         jg       G_M000_IG115
       EB08                 jmp      SHORT G_M000_IG123
 
G_M000_IG122:                ;; offset=0x0D20
       41BD01000000         mov      r13d, 1
       EB03                 jmp      SHORT G_M000_IG124
 
G_M000_IG123:                ;; offset=0x0D28
       4533ED               xor      r13d, r13d
 
G_M000_IG124:                ;; offset=0x0D2B
       48BFF80180295F7A0000 mov      rdi, 0x7A5F298001F8
       4C8B27               mov      r12, gword ptr [rdi]
       4180BC249D00000000   cmp      byte  ptr [r12+0x9D], 0
       0F84B9000000         je       G_M000_IG129
 
G_M000_IG125:                ;; offset=0x0D47
       488B8D30FFFFFF       mov      rcx, gword ptr [rbp-0xD0]
       83790800             cmp      dword ptr [rcx+0x08], 0
       0F84A8000000         je       G_M000_IG129
       488BF9               mov      rdi, rcx
       FF15CFD181FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       898568FFFFFF         mov      dword ptr [rbp-0x98], eax
       488BBD30FFFFFF       mov      rdi, gword ptr [rbp-0xD0]
       8B4F08               mov      ecx, dword ptr [rdi+0x08]
       898D64FFFFFF         mov      dword ptr [rbp-0x9C], ecx
       498BFF               mov      rdi, r15
       FF15B0D181FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       448BC0               mov      r8d, eax
       498BFC               mov      rdi, r12
       8B9568FFFFFF         mov      edx, dword ptr [rbp-0x98]
       8B8D64FFFFFF         mov      ecx, dword ptr [rbp-0x9C]
       BE03000000           mov      esi, 3
       FF157B63EFFF         call     [System.Diagnostics.Tracing.EventSource:WriteEvent(int,int,int,int):this]
       4585F5               test     r14d, r13d
       755E                 jne      SHORT G_M000_IG129
       488BBD30FFFFFF       mov      rdi, gword ptr [rbp-0xD0]
       FF1581D181FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       448BE8               mov      r13d, eax
       488BBD30FFFFFF       mov      rdi, gword ptr [rbp-0xD0]
       8B4708               mov      eax, dword ptr [rdi+0x08]
       898560FFFFFF         mov      dword ptr [rbp-0xA0], eax
       498BFF               mov      rdi, r15
       FF1565D181FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       4585F6               test     r14d, r14d
       750E                 jne      SHORT G_M000_IG126
       41B8FFFFFFFF         mov      r8d, -1
       41B901000000         mov      r9d, 1
       EB06                 jmp      SHORT G_M000_IG127
 
G_M000_IG126:                ;; offset=0x0DE0
       448BC3               mov      r8d, ebx
       4533C9               xor      r9d, r9d
 
G_M000_IG127:                ;; offset=0x0DE6
       498BFC               mov      rdi, r12
       418BF5               mov      esi, r13d
       8B9560FFFFFF         mov      edx, dword ptr [rbp-0xA0]
       FF157853DAFF         call     [System.Buffers.ArrayPoolEventSource:BufferDropped(int,int,int,int,int):this]
       EB06                 jmp      SHORT G_M000_IG129
 
G_M000_IG128:                ;; offset=0x0DFA
       E8D17BA8FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG129:                ;; offset=0x0E00
       4883C438             add      rsp, 56
       C3                   ret      
 
; Total bytes of code 3589
