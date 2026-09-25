; Assembly listing for method Lokad.Onnx.Tensor`1[float]:RunFloatMatMulKernel(int,int,int,ptr,ptr,ptr,Lokad.Onnx.TensorExecutionOptions) (Tier1)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Tier1 code
; optimized code
; optimized using Synthesized PGO
; rbp based frame
; fully interruptible
; with Synthesized PGO: fgCalledCount is 202624
; 13 inlinees with PGO data; 46 single block inlinees; 4 inlinees without PGO data

G_M000_IG01:                ;; offset=0x0000
       55                   push     rbp
       4157                 push     r15
       4156                 push     r14
       4155                 push     r13
       4154                 push     r12
       53                   push     rbx
       4881EC18010000       sub      rsp, 280
       C5F877               vzeroupper 
       488DAC2440010000     lea      rbp, [rsp+0x140]
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
       0F84B6090000         je       G_M000_IG79
 
G_M000_IG03:                ;; offset=0x0046
       440FB67D3D           movzx    r15, byte  ptr [rbp+0x3D]
       4585FF               test     r15d, r15d
       7409                 je       SHORT G_M000_IG04
       83FF01               cmp      edi, 1
       0F84F4020000         je       G_M000_IG30
 
G_M000_IG04:                ;; offset=0x0059
       4585FF               test     r15d, r15d
       0F848A090000         je       G_M000_IG78
       83FF02               cmp      edi, 2
       0F8C81090000         jl       G_M000_IG78
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
       0F84D8020000         je       G_M000_IG31
 
G_M000_IG05:                ;; offset=0x0095
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       837DAC40             cmp      dword ptr [rbp-0x54], 64
       0F8DD2030000         jge      G_M000_IG38
 
G_M000_IG06:                ;; offset=0x00A5
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       4585FF               test     r15d, r15d
       740D                 je       SHORT G_M000_IG08
 
G_M000_IG07:                ;; offset=0x00AD
       40F6C701             test     dil, 1
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       0F855C080000         jne      G_M000_IG69
 
G_M000_IG08:                ;; offset=0x00BA
       4863C6               movsxd   rax, esi
       448B75CC             mov      r14d, dword ptr [rbp-0x34]
       4963D6               movsxd   rdx, r14d
       480FAFC2             imul     rax, rdx
       483D00000100         cmp      rax, 0x10000
       0F8F42080000         jg       G_M000_IG69
 
G_M000_IG09:                ;; offset=0x00D4
       8BDE                 mov      ebx, esi
       410FAFDE             imul     ebx, r14d
       4C8B6D10             mov      r13, gword ptr [rbp+0x10]
       48B8682E8071AE7F0000 mov      rax, 0x7FAE71802E68
       4C8B20               mov      r12, gword ptr [rax]
       4C89A528FFFFFF       mov      gword ptr [rbp-0xD8], r12
       8BD3                 mov      edx, ebx
       49BAF8018071AE7F0000 mov      r10, 0x7FAE718001F8
       4D8B12               mov      r10, gword ptr [r10]
       4C899520FFFFFF       mov      gword ptr [rbp-0xE0], r10
       89559C               mov      dword ptr [rbp-0x64], edx
       448D5AFF             lea      r11d, [rdx-0x01]
       4183CB0F             or       r11d, 15
       F3450FBDDB           lzcnt    r11d, r11d
       4183F31F             xor      r11d, 31
       4183C3FD             add      r11d, -3
       44895D98             mov      dword ptr [rbp-0x68], r11d
       48BFE8ECE9D4B67F0000 mov      rdi, 0x7FB6D4E9ECE8
       48B8208881D5B67F0000 mov      rax, 0x7FB6D5818820
       FFD0                 call     rax
       833809               cmp      dword ptr [rax], 9
       0F8E32040000         jle      G_M000_IG45
       488B7808             mov      rdi, gword ptr [rax+0x08]
       488B4748             mov      rax, bword ptr [rdi+0x48]
       4885C0               test     rax, rax
       0F8421040000         je       G_M000_IG45
 
G_M000_IG10:                ;; offset=0x0154
       488B7810             mov      rdi, gword ptr [rax+0x10]
       4885FF               test     rdi, rdi
       0F847C040000         je       G_M000_IG48
       8B4708               mov      eax, dword ptr [rdi+0x08]
       8B4D98               mov      ecx, dword ptr [rbp-0x68]
       3BC1                 cmp      eax, ecx
       0F8671040000         jbe      G_M000_IG49
       8BC1                 mov      eax, ecx
       48C1E004             shl      rax, 4
       488B540710           mov      rdx, gword ptr [rdi+rax+0x10]
       4885D2               test     rdx, rdx
       0F8455040000         je       G_M000_IG47
       33F6                 xor      rsi, rsi
       4889740710           mov      gword ptr [rdi+rax+0x10], rsi
       488B8520FFFFFF       mov      rax, gword ptr [rbp-0xE0]
       80B89D00000000       cmp      byte  ptr [rax+0x9D], 0
       0F85E6030000         jne      G_M000_IG46
 
G_M000_IG11:                ;; offset=0x019E
       4C8BE2               mov      r12, rdx
 
G_M000_IG12:                ;; offset=0x01A1
       4D85ED               test     r13, r13
       7424                 je       SHORT G_M000_IG13
       48BE7083AF57B67F0000 mov      rsi, 0x7FB657AF8370
       49397500             cmp      qword ptr [r13], rsi
       0F8538070000         jne      G_M000_IG67
       4983C508             add      r13, 8
       4863F3               movsxd   rsi, ebx
       48C1E602             shl      rsi, 2
       F0                   lock     
       49017500             add      qword ptr [r13], rsi
 
G_M000_IG13:                ;; offset=0x01CA
       4C89A530FFFFFF       mov      gword ptr [rbp-0xD0], r12
 
G_M000_IG14:                ;; offset=0x01D1
       4C8965A0             mov      gword ptr [rbp-0x60], r12
       4D85E4               test     r12, r12
       7466                 je       SHORT G_M000_IG19
 
G_M000_IG15:                ;; offset=0x01DA
       41837C240800         cmp      dword ptr [r12+0x08], 0
       745E                 je       SHORT G_M000_IG19
       4983C410             add      r12, 16
       4D8BC4               mov      r8, r12
 
G_M000_IG16:                ;; offset=0x01E9
       418BFE               mov      edi, r14d
       C1FF1F               sar      edi, 31
       83E71F               and      edi, 31
       4103FE               add      edi, r14d
       83E7E0               and      edi, -32
       418BF6               mov      esi, r14d
       2BF7                 sub      esi, edi
       418BFE               mov      edi, r14d
       2BFE                 sub      edi, esi
       33F6                 xor      esi, esi
       85FF                 test     edi, edi
       7F3D                 jg       SHORT G_M000_IG20
 
G_M000_IG17:                ;; offset=0x0208
       418BF6               mov      esi, r14d
       2BF7                 sub      esi, edi
       8BD7                 mov      edx, edi
       C1FA1F               sar      edx, 31
       83E21F               and      edx, 31
       03D7                 add      edx, edi
       C1FA05               sar      edx, 5
       8B45D0               mov      eax, dword ptr [rbp-0x30]
       0FAFD0               imul     edx, eax
       C1E205               shl      edx, 5
       4863D2               movsxd   rdx, edx
       498D1490             lea      rdx, [r8+4*rdx]
       33C9                 xor      ecx, ecx
       85C0                 test     eax, eax
       0F8ED5000000         jle      G_M000_IG26
 
G_M000_IG18:                ;; offset=0x0234
       4863FF               movsxd   rdi, edi
       48C1E702             shl      rdi, 2
       E985000000           jmp      G_M000_IG24
 
G_M000_IG19:                ;; offset=0x0240
       4533C0               xor      r8d, r8d
       EBA4                 jmp      SHORT G_M000_IG16
                            align    [0 bytes for IG21]
 
G_M000_IG20:                ;; offset=0x0245
       8BD6                 mov      edx, esi
       C1FA1F               sar      edx, 31
       83E21F               and      edx, 31
       03D6                 add      edx, esi
       C1FA05               sar      edx, 5
       0FAF55D0             imul     edx, dword ptr [rbp-0x30]
       C1E205               shl      edx, 5
       4863D2               movsxd   rdx, edx
       498D1490             lea      rdx, [r8+4*rdx]
       33C9                 xor      ecx, ecx
       837DD000             cmp      dword ptr [rbp-0x30], 0
       7E48                 jle      SHORT G_M000_IG22
 
G_M000_IG21:                ;; offset=0x0268
       448BC9               mov      r9d, ecx
       41C1E105             shl      r9d, 5
       4D63C9               movsxd   r9, r9d
       4E8D0C8A             lea      r9, [rdx+4*r9]
       8BC1                 mov      eax, ecx
       410FAFC6             imul     eax, r14d
       4898                 cdqe     
       48C1E002             shl      rax, 2
       480345B8             add      rax, qword ptr [rbp-0x48]
       4C63D6               movsxd   r10, esi
       4A8D0490             lea      rax, [rax+4*r10]
       62F17E486F00         vmovdqu32 zmm0, zmmword ptr [rax]
       62F17E486F4801       vmovdqu32 zmm1, zmmword ptr [rax+0x40]
       62D17E487F01         vmovdqu32 zmmword ptr [r9], zmm0
       62D17E487F4901       vmovdqu32 zmmword ptr [r9+0x40], zmm1
       FFC1                 inc      ecx
       8B45D0               mov      eax, dword ptr [rbp-0x30]
       3BC8                 cmp      ecx, eax
       7CB8                 jl       SHORT G_M000_IG21
 
G_M000_IG22:                ;; offset=0x02B0
       8B45D0               mov      eax, dword ptr [rbp-0x30]
       83C620               add      esi, 32
       3BF7                 cmp      esi, edi
       7C8B                 jl       SHORT G_M000_IG20
       E949FFFFFF           jmp      G_M000_IG17
 
G_M000_IG23:                ;; offset=0x02BF
       FFC1                 inc      ecx
       3BC8                 cmp      ecx, eax
       7D44                 jge      SHORT G_M000_IG26
 
G_M000_IG24:                ;; offset=0x02C5
       448BC9               mov      r9d, ecx
       450FAFCE             imul     r9d, r14d
       4D63C9               movsxd   r9, r9d
       49C1E102             shl      r9, 2
       4C034DB8             add      r9, qword ptr [rbp-0x48]
       4C03CF               add      r9, rdi
       448BD1               mov      r10d, ecx
       440FAFD6             imul     r10d, esi
       4D63D2               movsxd   r10, r10d
       4E8D1492             lea      r10, [rdx+4*r10]
       4533DB               xor      r11d, r11d
       85F6                 test     esi, esi
       7ED0                 jle      SHORT G_M000_IG23
       90                   align    [1 bytes for IG25]
 
G_M000_IG25:                ;; offset=0x02F0
       4963DB               movsxd   rbx, r11d
       C4C17A100499         vmovss   xmm0, dword ptr [r9+4*rbx]
       C4C17A11049A         vmovss   dword ptr [r10+4*rbx], xmm0
       41FFC3               inc      r11d
       443BDE               cmp      r11d, esi
       7CE9                 jl       SHORT G_M000_IG25
       EBB6                 jmp      SHORT G_M000_IG23
 
G_M000_IG26:                ;; offset=0x0309
       4585FF               test     r15d, r15d
       7418                 je       SHORT G_M000_IG27
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8BF0                 mov      esi, eax
       418BD6               mov      edx, r14d
       488B4DC0             mov      rcx, qword ptr [rbp-0x40]
       4C8B4DB0             mov      r9, qword ptr [rbp-0x50]
       FF15E446DAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       EB17                 jmp      SHORT G_M000_IG28
 
G_M000_IG27:                ;; offset=0x0326
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8BF0                 mov      esi, eax
       418BD6               mov      edx, r14d
       488B4DC0             mov      rcx, qword ptr [rbp-0x40]
       4C8B4DB0             mov      r9, qword ptr [rbp-0x50]
       FF15B446DAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG28:                ;; offset=0x033D
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG29:                ;; offset=0x0343
       E869070000           call     G_M000_IG99
       E9F8050000           jmp      G_M000_IG72
 
G_M000_IG30:                ;; offset=0x034D
       817DCC00200000       cmp      dword ptr [rbp-0x34], 0x2000
       0F8CFFFCFFFF         jl       G_M000_IG04
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       BF01000000           mov      edi, 1
       FF156050DAFF         call     [Lokad.Onnx.MathOps:mm_m1_kblocked(int,int,int,ptr,ptr,ptr)]
       E9A3060000           jmp      G_M000_IG81
 
G_M000_IG31:                ;; offset=0x036D
       83FF40               cmp      edi, 64
       0F8C1FFDFFFF         jl       G_M000_IG05
       4863C6               movsxd   rax, esi
       486355CC             movsxd   rdx, dword ptr [rbp-0x34]
       480FAFC2             imul     rax, rdx
       483D00000004         cmp      rax, 0x4000000
       0F8F08FDFFFF         jg       G_M000_IG05
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
       448B75CC             mov      r14d, dword ptr [rbp-0x34]
       418BFE               mov      edi, r14d
       0FAF7DD0             imul     edi, dword ptr [rbp-0x30]
       FF159E4FDAFF         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       48898540FFFFFF       mov      gword ptr [rbp-0xC0], rax
 
G_M000_IG32:                ;; offset=0x0401
       488B8540FFFFFF       mov      rax, gword ptr [rbp-0xC0]
       488945A0             mov      gword ptr [rbp-0x60], rax
       4885C0               test     rax, rax
       7406                 je       SHORT G_M000_IG33
       83780800             cmp      dword ptr [rax+0x08], 0
       7504                 jne      SHORT G_M000_IG34
 
G_M000_IG33:                ;; offset=0x0417
       33DB                 xor      ebx, ebx
       EB04                 jmp      SHORT G_M000_IG35
 
G_M000_IG34:                ;; offset=0x041B
       488D5810             lea      rbx, bword ptr [rax+0x10]
 
G_M000_IG35:                ;; offset=0x041F
       8B7DD0               mov      edi, dword ptr [rbp-0x30]
       418BF6               mov      esi, r14d
       488B55B8             mov      rdx, qword ptr [rbp-0x48]
       488BCB               mov      rcx, rbx
       FF156EC7C7FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       418BD6               mov      edx, r14d
       488B4DC0             mov      rcx, qword ptr [rbp-0x40]
       4C8BC3               mov      r8, rbx
       4C8B4DB0             mov      r9, qword ptr [rbp-0x50]
       FF15A445DAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_3x4packed(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG36:                ;; offset=0x044D
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG37:                ;; offset=0x0453
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB540FFFFFF       mov      rsi, gword ptr [rbp-0xC0]
       33D2                 xor      edx, edx
       FF158919E4FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       E9EE040000           jmp      G_M000_IG73
 
G_M000_IG38:                ;; offset=0x0477
       4863C6               movsxd   rax, esi
       448B75CC             mov      r14d, dword ptr [rbp-0x34]
       4963D6               movsxd   rdx, r14d
       480FAFC2             imul     rax, rdx
       483D00000004         cmp      rax, 0x4000000
       0F8F14FCFFFF         jg       G_M000_IG06
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
       418BFE               mov      edi, r14d
       0FAF7DD0             imul     edi, dword ptr [rbp-0x30]
       FF159E4EDAFF         call     [Lokad.Onnx.Tensor`1[float]:RentScratch[float](int,Lokad.Onnx.TensorExecutionOptions):float[]]
       48898538FFFFFF       mov      gword ptr [rbp-0xC8], rax
 
G_M000_IG39:                ;; offset=0x0501
       488B8538FFFFFF       mov      rax, gword ptr [rbp-0xC8]
       488945A0             mov      gword ptr [rbp-0x60], rax
       4885C0               test     rax, rax
       7406                 je       SHORT G_M000_IG40
       83780800             cmp      dword ptr [rax+0x08], 0
       7505                 jne      SHORT G_M000_IG41
 
G_M000_IG40:                ;; offset=0x0517
       4533FF               xor      r15d, r15d
       EB04                 jmp      SHORT G_M000_IG42
 
G_M000_IG41:                ;; offset=0x051C
       4C8D7810             lea      r15, bword ptr [rax+0x10]
 
G_M000_IG42:                ;; offset=0x0520
       8B7DD0               mov      edi, dword ptr [rbp-0x30]
       418BF6               mov      esi, r14d
       488B55B8             mov      rdx, qword ptr [rbp-0x48]
       498BCF               mov      rcx, r15
       FF156DC6C7FF         call     [Lokad.Onnx.MathOps:PackPanelsB(int,int,ptr,ptr)]
       8B7DAC               mov      edi, dword ptr [rbp-0x54]
       8B75D0               mov      esi, dword ptr [rbp-0x30]
       418BD6               mov      edx, r14d
       488B4DC0             mov      rcx, qword ptr [rbp-0x40]
       4D8BC7               mov      r8, r15
       4C8B4DB0             mov      r9, qword ptr [rbp-0x50]
       FF15BB44DAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4packed_bump(int,int,int,ptr,ptr,ptr)]
       90                   nop      
 
G_M000_IG43:                ;; offset=0x054E
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG44:                ;; offset=0x0554
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB538FFFFFF       mov      rsi, gword ptr [rbp-0xC8]
       33D2                 xor      edx, edx
       FF158818E4FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       E9D0030000           jmp      G_M000_IG72
 
G_M000_IG45:                ;; offset=0x0575
       BF09000000           mov      edi, 9
       E8A156FEFF           call     CORINFO_HELP_GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED
       E9D0FBFFFF           jmp      G_M000_IG10
 
G_M000_IG46:                ;; offset=0x0584
       48899518FFFFFF       mov      gword ptr [rbp-0xE8], rdx
       488BFA               mov      rdi, rdx
       FF151CCE81FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       894580               mov      dword ptr [rbp-0x80], eax
       488B8D18FFFFFF       mov      rcx, gword ptr [rbp-0xE8]
       8B5108               mov      edx, dword ptr [rcx+0x08]
       89957CFFFFFF         mov      dword ptr [rbp-0x84], edx
       498BFC               mov      rdi, r12
       FF1500CE81FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       8B7580               mov      esi, dword ptr [rbp-0x80]
       8B957CFFFFFF         mov      edx, dword ptr [rbp-0x84]
       488BBD20FFFFFF       mov      rdi, gword ptr [rbp-0xE0]
       448B4598             mov      r8d, dword ptr [rbp-0x68]
       FF15CC50DAFF         call     [System.Buffers.ArrayPoolEventSource:BufferRented(int,int,int,int):this]
       488B9518FFFFFF       mov      rdx, gword ptr [rbp-0xE8]
       E9C6FBFFFF           jmp      G_M000_IG11
 
G_M000_IG47:                ;; offset=0x05D8
       8B4D98               mov      ecx, dword ptr [rbp-0x68]
       EB03                 jmp      SHORT G_M000_IG49
 
G_M000_IG48:                ;; offset=0x05DD
       8B4D98               mov      ecx, dword ptr [rbp-0x68]
 
G_M000_IG49:                ;; offset=0x05E0
       4C8BA528FFFFFF       mov      r12, gword ptr [rbp-0xD8]
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       394F08               cmp      dword ptr [rdi+0x08], ecx
       0F86EE010000         jbe      G_M000_IG62
       8BC1                 mov      eax, ecx
       488B7CC710           mov      rdi, gword ptr [rdi+8*rax+0x10]
       4885FF               test     rdi, rdi
       0F84CD010000         je       G_M000_IG61
       488B4708             mov      rax, gword ptr [rdi+0x08]
       48898510FFFFFF       mov      gword ptr [rbp-0xF0], rax
       48BFF008DD57B67F0000 mov      rdi, 0x7FB657DD08F0
       FF15E8B7A8FE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       803D1514AAFD00       cmp      byte  ptr [(reloc 0x7fb65583b35c)], 0
       740F                 je       SHORT G_M000_IG50
       E812EAA7FE           call     Interop+Sys:SchedGetCpu():int
       8BD0                 mov      edx, eax
       899570FFFFFF         mov      dword ptr [rbp-0x90], edx
       EB4B                 jmp      SHORT G_M000_IG52
 
G_M000_IG50:                ;; offset=0x0638
       BF0A000000           mov      edi, 10
       FF155553EFFF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B4010               mov      eax, dword ptr [rax+0x10]
       89856CFFFFFF         mov      dword ptr [rbp-0x94], eax
       BF0A000000           mov      edi, 10
       FF154153EFFF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B956CFFFFFF         mov      edx, dword ptr [rbp-0x94]
       8D4AFF               lea      ecx, [rdx-0x01]
       894810               mov      dword ptr [rax+0x10], ecx
       0FB7C2               movzx    rax, dx
       85C0                 test     eax, eax
       7510                 jne      SHORT G_M000_IG51
       FF154053EFFF         call     [System.Threading.ProcessorIdCache:RefreshCurrentProcessorId():int]
       8BD0                 mov      edx, eax
       899570FFFFFF         mov      dword ptr [rbp-0x90], edx
       EB09                 jmp      SHORT G_M000_IG52
 
G_M000_IG51:                ;; offset=0x067A
       C1FA10               sar      edx, 16
       899570FFFFFF         mov      dword ptr [rbp-0x90], edx
 
G_M000_IG52:                ;; offset=0x0683
       48BF9007DD57B67F0000 mov      rdi, 0x7FB657DD0790
       FF1575B7A8FE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       8B8570FFFFFF         mov      eax, dword ptr [rbp-0x90]
       33D2                 xor      edx, edx
       F7358F13AAFD         div      edx:eax, dword ptr [(reloc 0x7fb65583b350)]
       8BC2                 mov      eax, edx
       33C9                 xor      ecx, ecx
       E99E000000           jmp      G_M000_IG56
 
G_M000_IG53:                ;; offset=0x06AA
       3B4208               cmp      eax, dword ptr [rdx+0x08]
       0F8370030000         jae      G_M000_IG83
       898578FFFFFF         mov      dword ptr [rbp-0x88], eax
       8BF8                 mov      edi, eax
       488B74FA10           mov      rsi, gword ptr [rdx+8*rdi+0x10]
       4889B500FFFFFF       mov      gword ptr [rbp-0x100], rsi
       403836               cmp      byte  ptr [rsi], sil
       4533C0               xor      r8, r8
       4C898508FFFFFF       mov      gword ptr [rbp-0xF8], r8
       488BFE               mov      rdi, rsi
       FF15A3CF0400         call     [System.Threading.Monitor:Enter(System.Object)]
       488BB500FFFFFF       mov      rsi, gword ptr [rbp-0x100]
       488B7E08             mov      rdi, gword ptr [rsi+0x08]
       8B4610               mov      eax, dword ptr [rsi+0x10]
       FFC8                 dec      eax
       394708               cmp      dword ptr [rdi+0x08], eax
       761B                 jbe      SHORT G_M000_IG54
       8BC8                 mov      ecx, eax
       4C8B44CF10           mov      r8, gword ptr [rdi+8*rcx+0x10]
       4C898508FFFFFF       mov      gword ptr [rbp-0xF8], r8
       8BD0                 mov      edx, eax
       4533C0               xor      r8, r8
       4C8944D710           mov      gword ptr [rdi+8*rdx+0x10], r8
       894610               mov      dword ptr [rsi+0x10], eax
 
G_M000_IG54:                ;; offset=0x070D
       488BFE               mov      rdi, rsi
       FF15D2C7A8FE         call     [System.Threading.Monitor:Exit(System.Object)]
       488B8D08FFFFFF       mov      rcx, gword ptr [rbp-0xF8]
       4885C9               test     rcx, rcx
       753E                 jne      SHORT G_M000_IG57
       8B8578FFFFFF         mov      eax, dword ptr [rbp-0x88]
       FFC0                 inc      eax
       8BC8                 mov      ecx, eax
       488BBD10FFFFFF       mov      rdi, gword ptr [rbp-0xF0]
       394F08               cmp      dword ptr [rdi+0x08], ecx
       7502                 jne      SHORT G_M000_IG55
       33C9                 xor      ecx, ecx
 
G_M000_IG55:                ;; offset=0x073A
       8B8574FFFFFF         mov      eax, dword ptr [rbp-0x8C]
       FFC0                 inc      eax
       8BD0                 mov      edx, eax
       8BC1                 mov      eax, ecx
       8BCA                 mov      ecx, edx
 
G_M000_IG56:                ;; offset=0x0748
       488B9510FFFFFF       mov      rdx, gword ptr [rbp-0xF0]
       898D74FFFFFF         mov      dword ptr [rbp-0x8C], ecx
       394A08               cmp      dword ptr [rdx+0x08], ecx
       0F8F4CFFFFFF         jg       G_M000_IG53
       EB02                 jmp      SHORT G_M000_IG58
 
G_M000_IG57:                ;; offset=0x0760
       EB02                 jmp      SHORT G_M000_IG59
 
G_M000_IG58:                ;; offset=0x0762
       33C9                 xor      rcx, rcx
 
G_M000_IG59:                ;; offset=0x0764
       488BC1               mov      rax, rcx
       4885C0               test     rax, rax
       7466                 je       SHORT G_M000_IG61
       488B8D20FFFFFF       mov      rcx, gword ptr [rbp-0xE0]
       80B99D00000000       cmp      byte  ptr [rcx+0x9D], 0
       744E                 je       SHORT G_M000_IG60
       48898518FFFFFF       mov      gword ptr [rbp-0xE8], rax
       488BF8               mov      rdi, rax
       FF1524CC81FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       894588               mov      dword ptr [rbp-0x78], eax
       488B8D18FFFFFF       mov      rcx, gword ptr [rbp-0xE8]
       8B5108               mov      edx, dword ptr [rcx+0x08]
       895584               mov      dword ptr [rbp-0x7C], edx
       498BFC               mov      rdi, r12
       FF150BCC81FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       8B7588               mov      esi, dword ptr [rbp-0x78]
       8B5584               mov      edx, dword ptr [rbp-0x7C]
       488BBD20FFFFFF       mov      rdi, gword ptr [rbp-0xE0]
       448B4598             mov      r8d, dword ptr [rbp-0x68]
       FF15DA4EDAFF         call     [System.Buffers.ArrayPoolEventSource:BufferRented(int,int,int,int):this]
       488B9518FFFFFF       mov      rdx, gword ptr [rbp-0xE8]
       E9D4F9FFFF           jmp      G_M000_IG11
 
G_M000_IG60:                ;; offset=0x07CA
       488BD0               mov      rdx, rax
       E9CCF9FFFF           jmp      G_M000_IG11
 
G_M000_IG61:                ;; offset=0x07D2
       BF10000000           mov      edi, 16
       8B4D98               mov      ecx, dword ptr [rbp-0x68]
       C4E271F7D7           shlx     edx, edi, ecx
       8BC2                 mov      eax, edx
       EB2E                 jmp      SHORT G_M000_IG64
 
G_M000_IG62:                ;; offset=0x07E3
       8B459C               mov      eax, dword ptr [rbp-0x64]
       85C0                 test     eax, eax
       750F                 jne      SHORT G_M000_IG63
       49BCE0AEC045B67F0000 mov      r12, 0x7FB645C0AEE0
       E9A8F9FFFF           jmp      G_M000_IG12
 
G_M000_IG63:                ;; offset=0x07F9
       89459C               mov      dword ptr [rbp-0x64], eax
       8BF8                 mov      edi, eax
       48BEF810C045B67F0000 mov      rsi, 0x7FB645C010F8
       FF15FA2323FF         call     [System.ArgumentOutOfRangeException:ThrowIfNegative[int](int,System.String)]
       8B459C               mov      eax, dword ptr [rbp-0x64]
 
G_M000_IG64:                ;; offset=0x0811
       3D00020000           cmp      eax, 512
       7D17                 jge      SHORT G_M000_IG65
       4863F0               movsxd   rsi, eax
       48BF788F9256B67F0000 mov      rdi, 0x7FB656928F78
       E80609067D           call     CORINFO_HELP_NEWARR_1_VC
       488BC8               mov      rcx, rax
       EB0D                 jmp      SHORT G_M000_IG66
 
G_M000_IG65:                ;; offset=0x082F
       8BF8                 mov      edi, eax
       33F6                 xor      esi, esi
       FF15BF4EDAFF         call     [System.GC:<AllocateUninitializedArray>g__AllocateNewArrayWorker|77_0[float](int,bool):float[]]
       488BC8               mov      rcx, rax
 
G_M000_IG66:                ;; offset=0x083C
       488BC1               mov      rax, rcx
       488B8D20FFFFFF       mov      rcx, gword ptr [rbp-0xE0]
       80B99D00000000       cmp      byte  ptr [rcx+0x9D], 0
       0F84BB000000         je       G_M000_IG68
       3800                 cmp      byte  ptr [rax], al
       48898518FFFFFF       mov      gword ptr [rbp-0xE8], rax
       488BF8               mov      rdi, rax
       FF154BCB81FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       894594               mov      dword ptr [rbp-0x6C], eax
       488B8D18FFFFFF       mov      rcx, gword ptr [rbp-0xE8]
       8B5108               mov      edx, dword ptr [rcx+0x08]
       895590               mov      dword ptr [rbp-0x70], edx
       498BFC               mov      rdi, r12
       FF1532CB81FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       8B7594               mov      esi, dword ptr [rbp-0x6C]
       8B5590               mov      edx, dword ptr [rbp-0x70]
       488BBD20FFFFFF       mov      rdi, gword ptr [rbp-0xE0]
       41B8FFFFFFFF         mov      r8d, -1
       FF15FF4DDAFF         call     [System.Buffers.ArrayPoolEventSource:BufferRented(int,int,int,int):this]
       488B8518FFFFFF       mov      rax, gword ptr [rbp-0xE8]
       8B4808               mov      ecx, dword ptr [rax+0x08]
       894D8C               mov      dword ptr [rbp-0x74], ecx
       498BFC               mov      rdi, r12
       FF1501CB81FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       498B7C2410           mov      rdi, gword ptr [r12+0x10]
       41B901000000         mov      r9d, 1
       BE02000000           mov      esi, 2
       448B6598             mov      r12d, dword ptr [rbp-0x68]
       44396708             cmp      dword ptr [rdi+0x08], r12d
       440F4FCE             cmovg    r9d, esi
       488BBD20FFFFFF       mov      rdi, gword ptr [rbp-0xE0]
       8B7594               mov      esi, dword ptr [rbp-0x6C]
       8B558C               mov      edx, dword ptr [rbp-0x74]
       41B8FFFFFFFF         mov      r8d, -1
       FF15FA4DDAFF         call     [System.Buffers.ArrayPoolEventSource:BufferAllocated(int,int,int,int,int):this]
       488B9518FFFFFF       mov      rdx, gword ptr [rbp-0xE8]
       E9ACF8FFFF           jmp      G_M000_IG11
 
G_M000_IG67:                ;; offset=0x08F2
       4863F3               movsxd   rsi, ebx
       48C1E602             shl      rsi, 2
       498BFD               mov      rdi, r13
       49BBC8248455B67F0000 mov      r11, 0x7FB6558424C8
       41FF13               call     [r11]Lokad.Onnx.IScratchAccountant:AddScratchBytes(long):this
       E9BCF8FFFF           jmp      G_M000_IG13
 
G_M000_IG68:                ;; offset=0x090E
       488BD0               mov      rdx, rax
       E988F8FFFF           jmp      G_M000_IG11
 
G_M000_IG69:                ;; offset=0x0916
       448B75CC             mov      r14d, dword ptr [rbp-0x34]
       81FE000A0000         cmp      esi, 0xA00
       7D17                 jge      SHORT G_M000_IG71
 
G_M000_IG70:                ;; offset=0x0922
       4181FE000A0000       cmp      r14d, 0xA00
       7D0E                 jge      SHORT G_M000_IG71
       8B7DAC               mov      edi, dword ptr [rbp-0x54]
       418BD6               mov      edx, r14d
       FF15A94ADAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4tiled(int,int,int,ptr,ptr,ptr)]
       EB0C                 jmp      SHORT G_M000_IG72
 
G_M000_IG71:                ;; offset=0x0939
       8B7DAC               mov      edi, dword ptr [rbp-0x54]
       418BD6               mov      edx, r14d
       FF15B34ADAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics_2x4(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG72:                ;; offset=0x0945
       41B9ABAAAAAA         mov      r9d, 0xAAAAAAAB
       8B4DD4               mov      ecx, dword ptr [rbp-0x2C]
       4C0FAFC9             imul     r9, rcx
       49C1E921             shr      r9, 33
       478D0C49             lea      r9d, [r9+2*r9]
       8B7DD4               mov      edi, dword ptr [rbp-0x2C]
       448BFF               mov      r15d, edi
       452BF9               sub      r15d, r9d
       753F                 jne      SHORT G_M000_IG75
 
G_M000_IG73:                ;; offset=0x0965
       4C634DD0             movsxd   r9, dword ptr [rbp-0x30]
       48634DCC             movsxd   rcx, dword ptr [rbp-0x34]
       4C0FAFC9             imul     r9, rcx
       4981F900000004       cmp      r9, 0x4000000
       7F2A                 jg       SHORT G_M000_IG75
       83FF40               cmp      edi, 64
       7D1D                 jge      SHORT G_M000_IG74
       4C634DD0             movsxd   r9, dword ptr [rbp-0x30]
       48634DCC             movsxd   rcx, dword ptr [rbp-0x34]
       4C0FAFC9             imul     r9, rcx
       4981F900000100       cmp      r9, 0x10000
       410F9EC1             setle    r9b
       450FB6C9             movzx    r9, r9b
       EB0B                 jmp      SHORT G_M000_IG76
 
G_M000_IG74:                ;; offset=0x099C
       41B901000000         mov      r9d, 1
       EB03                 jmp      SHORT G_M000_IG76
 
G_M000_IG75:                ;; offset=0x09A4
       4533C9               xor      r9d, r9d
 
G_M000_IG76:                ;; offset=0x09A7
       397DAC               cmp      dword ptr [rbp-0x54], edi
       7464                 je       SHORT G_M000_IG81
 
G_M000_IG77:                ;; offset=0x09AC
       4585C9               test     r9d, r9d
       755F                 jne      SHORT G_M000_IG81
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
       FF15264ADAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       EB24                 jmp      SHORT G_M000_IG81
 
G_M000_IG78:                ;; offset=0x09EC
       4585FF               test     r15d, r15d
       7416                 je       SHORT G_M000_IG80
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       FF15164ADAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized_intrinsics(int,int,int,ptr,ptr,ptr)]
       EB14                 jmp      SHORT G_M000_IG81
 
G_M000_IG79:                ;; offset=0x09FC
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       FF153B4ADAFF         call     [Lokad.Onnx.MathOps:mm(int,int,int,ptr,ptr,ptr)]
       EB09                 jmp      SHORT G_M000_IG81
 
G_M000_IG80:                ;; offset=0x0A07
       8B55CC               mov      edx, dword ptr [rbp-0x34]
       FF15184ADAFF         call     [Lokad.Onnx.MathOps:mm_unsafe_vectorized(int,int,int,ptr,ptr,ptr)]
 
G_M000_IG81:                ;; offset=0x0A10
       90                   nop      
 
G_M000_IG82:                ;; offset=0x0A11
       4881C418010000       add      rsp, 280
       5B                   pop      rbx
       415C                 pop      r12
       415D                 pop      r13
       415E                 pop      r14
       415F                 pop      r15
       5D                   pop      rbp
       C3                   ret      
 
G_M000_IG83:                ;; offset=0x0A23
       E82874A8FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG84:                ;; offset=0x0A29
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG85:                ;; offset=0x0A30
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG86:                ;; offset=0x0A36
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG87:                ;; offset=0x0A3B
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG88:                ;; offset=0x0A42
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB540FFFFFF       mov      rsi, gword ptr [rbp-0xC0]
       33D2                 xor      edx, edx
       FF159A13E4FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       90                   nop      
 
G_M000_IG89:                ;; offset=0x0A5F
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG90:                ;; offset=0x0A64
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG91:                ;; offset=0x0A6B
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG92:                ;; offset=0x0A71
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG93:                ;; offset=0x0A76
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG94:                ;; offset=0x0A7D
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       488B3F               mov      rdi, gword ptr [rdi]
       488BB538FFFFFF       mov      rsi, gword ptr [rbp-0xC8]
       33D2                 xor      edx, edx
       FF155F13E4FF         call     [System.Buffers.SharedArrayPool`1[float]:Return(float[],bool):this]
       90                   nop      
 
G_M000_IG95:                ;; offset=0x0A9A
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG96:                ;; offset=0x0A9F
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG97:                ;; offset=0x0AA6
       33FF                 xor      rdi, rdi
       48897DA0             mov      gword ptr [rbp-0x60], rdi
 
G_M000_IG98:                ;; offset=0x0AAC
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG99:                ;; offset=0x0AB1
       4883EC38             sub      rsp, 56
       C5F877               vzeroupper 
 
G_M000_IG100:                ;; offset=0x0AB8
       48BF682E8071AE7F0000 mov      rdi, 0x7FAE71802E68
       4C8B3F               mov      r15, gword ptr [rdi]
       4883BD30FFFFFF00     cmp      gword ptr [rbp-0xD0], 0
       750C                 jne      SHORT G_M000_IG102
 
G_M000_IG101:                ;; offset=0x0ACF
       BF02000000           mov      edi, 2
       FF152E4BBCFF         call     [System.ThrowHelper:ThrowArgumentNullException(int)]
       CC                   int3     
 
G_M000_IG102:                ;; offset=0x0ADB
       488BBD30FFFFFF       mov      rdi, gword ptr [rbp-0xD0]
       8B4708               mov      eax, dword ptr [rdi+0x08]
       FFC8                 dec      eax
       83C80F               or       eax, 15
       33DB                 xor      ebx, ebx
       F30FBDD8             lzcnt    ebx, eax
       83F31F               xor      ebx, 31
       83C3FD               add      ebx, -3
       48BFE8ECE9D4B67F0000 mov      rdi, 0x7FB6D4E9ECE8
       48B8208881D5B67F0000 mov      rax, 0x7FB6D5818820
       FFD0                 call     rax
       833809               cmp      dword ptr [rax], 9
       7E0D                 jle      SHORT G_M000_IG103
       488B7808             mov      rdi, gword ptr [rax+0x08]
       488B4748             mov      rax, bword ptr [rdi+0x48]
       4885C0               test     rax, rax
       750A                 jne      SHORT G_M000_IG104
 
G_M000_IG103:                ;; offset=0x0B1E
       BF09000000           mov      edi, 9
       E8F850FEFF           call     CORINFO_HELP_GETDYNAMIC_GCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED
 
G_M000_IG104:                ;; offset=0x0B28
       488B4010             mov      rax, gword ptr [rax+0x10]
       4885C0               test     rax, rax
       7509                 jne      SHORT G_M000_IG106
 
G_M000_IG105:                ;; offset=0x0B31
       498BFF               mov      rdi, r15
       FF157E4CDAFF         call     [System.Buffers.SharedArrayPool`1[float]:InitializeTlsBucketsAndTrimming():System.Buffers.SharedArrayPoolThreadLocalArray[]:this]
 
G_M000_IG106:                ;; offset=0x0B3A
       4533F6               xor      r14d, r14d
       41BD01000000         mov      r13d, 1
       395808               cmp      dword ptr [rax+0x08], ebx
       0F862B020000         jbe      G_M000_IG124
 
G_M000_IG107:                ;; offset=0x0B4C
       41BE01000000         mov      r14d, 1
       BF10000000           mov      edi, 16
       C4E261F7FF           shlx     edi, edi, ebx
       488B8D30FFFFFF       mov      rcx, gword ptr [rbp-0xD0]
       397908               cmp      dword ptr [rcx+0x08], edi
       7448                 je       SHORT G_M000_IG109
 
G_M000_IG108:                ;; offset=0x0B68
       48BF28CD8656B67F0000 mov      rdi, 0x7FB65686CD28
       E8A904067D           call     CORINFO_HELP_NEWSFAST
       4C8BF8               mov      r15, rax
       FF15504CDAFF         call     [System.SR:get_ArgumentException_BufferNotFromPool():System.String]
       4C8BF0               mov      r14, rax
       BF6D040000           mov      edi, 0x46D
       48BE00408355B67F0000 mov      rsi, 0x7FB655834000
       FF15509F6FFF         call     [CORINFO_HELP_STRCNS]
       488BD0               mov      rdx, rax
       498BF6               mov      rsi, r14
       498BFF               mov      rdi, r15
       FF1561A06FFF         call     [System.ArgumentException:.ctor(System.String,System.String):this]
       498BFF               mov      rdi, r15
       E81131F17C           call     CORINFO_HELP_THROW
       CC                   int3     
 
G_M000_IG109:                ;; offset=0x0BB0
       8BFB                 mov      edi, ebx
       48C1E704             shl      rdi, 4
       4C8D643810           lea      r12, bword ptr [rax+rdi+0x10]
       498B0424             mov      rax, gword ptr [r12]
       488985F8FEFFFF       mov      gword ptr [rbp-0x108], rax
       498BFC               mov      rdi, r12
       488BF1               mov      rsi, rcx
       E84F5BA8FD           call     CORINFO_HELP_ASSIGN_REF
       33FF                 xor      edi, edi
       41897C2408           mov      dword ptr [r12+0x08], edi
       4C8BA5F8FEFFFF       mov      r12, gword ptr [rbp-0x108]
       4D85E4               test     r12, r12
       0F848F010000         je       G_M000_IG124
 
G_M000_IG110:                ;; offset=0x0BE8
       498B7F10             mov      rdi, gword ptr [r15+0x10]
       3B5F08               cmp      ebx, dword ptr [rdi+0x08]
       0F8351020000         jae      G_M000_IG128
       8BF3                 mov      esi, ebx
       488B44F710           mov      rax, gword ptr [rdi+8*rsi+0x10]
       4885C0               test     rax, rax
       750B                 jne      SHORT G_M000_IG111
       498BFF               mov      rdi, r15
       8BF3                 mov      esi, ebx
       FF15F44BDAFF         call     [System.Buffers.SharedArrayPool`1[float]:CreatePerCorePartitions(int):System.Buffers.SharedArrayPoolPartitions:this]
 
G_M000_IG111:                ;; offset=0x0C0C
       4C8B6808             mov      r13, gword ptr [rax+0x08]
       48BFF008DD57B67F0000 mov      rdi, 0x7FB657DD08F0
       FF15E8B1A8FE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       803D150EAAFD00       cmp      byte  ptr [(reloc 0x7fb65583b35c)], 0
       740F                 je       SHORT G_M000_IG112
       E812E4A7FE           call     Interop+Sys:SchedGetCpu():int
       8BD0                 mov      edx, eax
       899554FFFFFF         mov      dword ptr [rbp-0xAC], edx
       EB4B                 jmp      SHORT G_M000_IG114
 
G_M000_IG112:                ;; offset=0x0C38
       BF0A000000           mov      edi, 10
       FF15554DEFFF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B4010               mov      eax, dword ptr [rax+0x10]
       898550FFFFFF         mov      dword ptr [rbp-0xB0], eax
       BF0A000000           mov      edi, 10
       FF15414DEFFF         call     [CORINFO_HELP_GETDYNAMIC_NONGCTHREADSTATIC_BASE_NOCTOR_OPTIMIZED]
       8B9550FFFFFF         mov      edx, dword ptr [rbp-0xB0]
       8D4AFF               lea      ecx, [rdx-0x01]
       894810               mov      dword ptr [rax+0x10], ecx
       0FB7C2               movzx    rax, dx
       85C0                 test     eax, eax
       7510                 jne      SHORT G_M000_IG113
       FF15404DEFFF         call     [System.Threading.ProcessorIdCache:RefreshCurrentProcessorId():int]
       8BD0                 mov      edx, eax
       899554FFFFFF         mov      dword ptr [rbp-0xAC], edx
       EB09                 jmp      SHORT G_M000_IG114
 
G_M000_IG113:                ;; offset=0x0C7A
       C1FA10               sar      edx, 16
       899554FFFFFF         mov      dword ptr [rbp-0xAC], edx
 
G_M000_IG114:                ;; offset=0x0C83
       48BF9007DD57B67F0000 mov      rdi, 0x7FB657DD0790
       FF1575B1A8FE         call     [CORINFO_HELP_GET_NONGCSTATIC_BASE]
       8B8554FFFFFF         mov      eax, dword ptr [rbp-0xAC]
       33D2                 xor      edx, edx
       F7358F0DAAFD         div      edx:eax, dword ptr [(reloc 0x7fb65583b350)]
       8BC2                 mov      eax, edx
       33C9                 xor      ecx, ecx
       E9B0000000           jmp      G_M000_IG121
 
G_M000_IG115:                ;; offset=0x0CAA
       413B4508             cmp      eax, dword ptr [r13+0x08]
       0F8392010000         jae      G_M000_IG128
       89855CFFFFFF         mov      dword ptr [rbp-0xA4], eax
       8BF8                 mov      edi, eax
       498B54FD10           mov      rdx, gword ptr [r13+8*rdi+0x10]
       488995F0FEFFFF       mov      gword ptr [rbp-0x110], rdx
       3812                 cmp      byte  ptr [rdx], dl
       33F6                 xor      esi, esi
       89B54CFFFFFF         mov      dword ptr [rbp-0xB4], esi
       488BFA               mov      rdi, rdx
       FF15A5C90400         call     [System.Threading.Monitor:Enter(System.Object)]
       488B85F0FEFFFF       mov      rax, gword ptr [rbp-0x110]
       488B7808             mov      rdi, gword ptr [rax+0x08]
       8B4810               mov      ecx, dword ptr [rax+0x10]
       898D48FFFFFF         mov      dword ptr [rbp-0xB8], ecx
       394F08               cmp      dword ptr [rdi+0x08], ecx
       7634                 jbe      SHORT G_M000_IG117
       85C9                 test     ecx, ecx
       7544                 jne      SHORT G_M000_IG118
       33F6                 xor      esi, esi
       897014               mov      dword ptr [rax+0x14], esi
 
G_M000_IG116:                ;; offset=0x0CFD
       8BF1                 mov      esi, ecx
       488D7CF710           lea      rdi, bword ptr [rdi+8*rsi+0x10]
       498BF4               mov      rsi, r12
       E8145AA8FD           call     CORINFO_HELP_ASSIGN_REF
       8BBD48FFFFFF         mov      edi, dword ptr [rbp-0xB8]
       FFC7                 inc      edi
       488B85F0FEFFFF       mov      rax, gword ptr [rbp-0x110]
       897810               mov      dword ptr [rax+0x10], edi
       C7854CFFFFFF01000000 mov      dword ptr [rbp-0xB4], 1
 
G_M000_IG117:                ;; offset=0x0D28
       488BF8               mov      rdi, rax
       FF15B7C1A8FE         call     [System.Threading.Monitor:Exit(System.Object)]
       83BD4CFFFFFF00       cmp      dword ptr [rbp-0xB4], 0
       7404                 je       SHORT G_M000_IG119
       EB30                 jmp      SHORT G_M000_IG122
 
G_M000_IG118:                ;; offset=0x0D3C
       EBBF                 jmp      SHORT G_M000_IG116
 
G_M000_IG119:                ;; offset=0x0D3E
       8B855CFFFFFF         mov      eax, dword ptr [rbp-0xA4]
       FFC0                 inc      eax
       8BF8                 mov      edi, eax
       41397D08             cmp      dword ptr [r13+0x08], edi
       7502                 jne      SHORT G_M000_IG120
       33FF                 xor      edi, edi
 
G_M000_IG120:                ;; offset=0x0D50
       8B8D58FFFFFF         mov      ecx, dword ptr [rbp-0xA8]
       FFC1                 inc      ecx
       8BC7                 mov      eax, edi
 
G_M000_IG121:                ;; offset=0x0D5A
       898D58FFFFFF         mov      dword ptr [rbp-0xA8], ecx
       41394D08             cmp      dword ptr [r13+0x08], ecx
       0F8F40FFFFFF         jg       G_M000_IG115
       EB08                 jmp      SHORT G_M000_IG123
 
G_M000_IG122:                ;; offset=0x0D6C
       41BD01000000         mov      r13d, 1
       EB03                 jmp      SHORT G_M000_IG124
 
G_M000_IG123:                ;; offset=0x0D74
       4533ED               xor      r13d, r13d
 
G_M000_IG124:                ;; offset=0x0D77
       48BFF8018071AE7F0000 mov      rdi, 0x7FAE718001F8
       4C8B27               mov      r12, gword ptr [rdi]
       4180BC249D00000000   cmp      byte  ptr [r12+0x9D], 0
       0F84B9000000         je       G_M000_IG129
 
G_M000_IG125:                ;; offset=0x0D93
       488B8D30FFFFFF       mov      rcx, gword ptr [rbp-0xD0]
       83790800             cmp      dword ptr [rcx+0x08], 0
       0F84A8000000         je       G_M000_IG129
       488BF9               mov      rdi, rcx
       FF1503C681FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       898568FFFFFF         mov      dword ptr [rbp-0x98], eax
       488BBD30FFFFFF       mov      rdi, gword ptr [rbp-0xD0]
       8B4F08               mov      ecx, dword ptr [rdi+0x08]
       898D64FFFFFF         mov      dword ptr [rbp-0x9C], ecx
       498BFF               mov      rdi, r15
       FF15E4C581FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       448BC0               mov      r8d, eax
       498BFC               mov      rdi, r12
       8B9568FFFFFF         mov      edx, dword ptr [rbp-0x98]
       8B8D64FFFFFF         mov      ecx, dword ptr [rbp-0x9C]
       BE03000000           mov      esi, 3
       FF159FA20400         call     [System.Diagnostics.Tracing.EventSource:WriteEvent(int,int,int,int):this]
       4585F5               test     r14d, r13d
       755E                 jne      SHORT G_M000_IG129
       488BBD30FFFFFF       mov      rdi, gword ptr [rbp-0xD0]
       FF15B5C581FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       448BE8               mov      r13d, eax
       488BBD30FFFFFF       mov      rdi, gword ptr [rbp-0xD0]
       8B4708               mov      eax, dword ptr [rdi+0x08]
       898560FFFFFF         mov      dword ptr [rbp-0xA0], eax
       498BFF               mov      rdi, r15
       FF1599C581FF         call     [System.Runtime.CompilerServices.RuntimeHelpers:GetHashCode(System.Object):int]
       8BC8                 mov      ecx, eax
       4585F6               test     r14d, r14d
       750E                 jne      SHORT G_M000_IG126
       41B8FFFFFFFF         mov      r8d, -1
       41B901000000         mov      r9d, 1
       EB06                 jmp      SHORT G_M000_IG127
 
G_M000_IG126:                ;; offset=0x0E2C
       448BC3               mov      r8d, ebx
       4533C9               xor      r9d, r9d
 
G_M000_IG127:                ;; offset=0x0E32
       498BFC               mov      rdi, r12
       418BF5               mov      esi, r13d
       8B9560FFFFFF         mov      edx, dword ptr [rbp-0xA0]
       FF15044ADAFF         call     [System.Buffers.ArrayPoolEventSource:BufferDropped(int,int,int,int,int):this]
       EB06                 jmp      SHORT G_M000_IG129
 
G_M000_IG128:                ;; offset=0x0E46
       E80570A8FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG129:                ;; offset=0x0E4C
       4883C438             add      rsp, 56
       C3                   ret      
 
; Total bytes of code 3665
