; Assembly listing for method Lokad.Onnx.Tensor`1[float]:RunBatchedFloatMatMul(Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],Lokad.Onnx.TensorExecutionOptions) (Instrumented Tier0)
; Emitting BLENDED_CODE for generic X64 + VEX + EVEX on Unix
; Instrumented Tier0 code
; rbp based frame
; fully interruptible
; compiling with minopt

G_M000_IG01:                ;; offset=0x0000
       55                   push     rbp
       4881EC20030000       sub      rsp, 800
       488DAC2420030000     lea      rbp, [rsp+0x320]
       C4413857C0           vxorps   xmm8, xmm8, xmm8
       62717D087F45D1       vmovdqa32 xmmword ptr [rbp-0x2F0], xmm8
       48B860FDFFFFFFFFFFFF mov      rax, -672
       C5797F4405C0         vmovdqa  xmmword ptr [rbp+rax-0x40], xmm8
       C5797F4405D0         vmovdqa  xmmword ptr [rbp+rax-0x30], xmm8
       C5797F4405E0         vmovdqa  xmmword ptr [rbp+rax-0x20], xmm8
       4883C030             add      rax, 48
       75E8                 jne      SHORT  -5 instr
       48897DD0             mov      gword ptr [rbp-0x30], rdi
       488975C8             mov      gword ptr [rbp-0x38], rsi
       488955C0             mov      gword ptr [rbp-0x40], rdx
 
G_M000_IG02:                ;; offset=0x004A
       C78548FDFFFFE8030000 mov      dword ptr [rbp-0x2B8], 0x3E8
       48BF50065F6B947C0000 mov      rdi, 0x7C946B5F0650
       E83D3A337D           call     CORINFO_HELP_NEWSFAST
       488985C0FEFFFF       mov      gword ptr [rbp-0x140], rax
       488BBDC0FEFFFF       mov      rdi, gword ptr [rbp-0x140]
       FF15D96F0900         call     [Lokad.Onnx.Tensor`1+<>c__DisplayClass434_0[float]:.ctor():this]
       488B85C0FEFFFF       mov      rax, gword ptr [rbp-0x140]
       488945B8             mov      gword ptr [rbp-0x48], rax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488D7848             lea      rdi, bword ptr [rax+0x48]
       488D7510             lea      rsi, bword ptr [rbp+0x10]
       E87D29337D           call     CORINFO_HELP_ASSIGN_BYREF
       E87829337D           call     CORINFO_HELP_ASSIGN_BYREF
       E87329337D           call     CORINFO_HELP_ASSIGN_BYREF
       E86E29337D           call     CORINFO_HELP_ASSIGN_BYREF
       E86929337D           call     CORINFO_HELP_ASSIGN_BYREF
       48A5                 movsq    
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       3800                 cmp      byte  ptr [rax], al
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488D7848             lea      rdi, bword ptr [rax+0x48]
       FF15F35A0900         call     [Lokad.Onnx.TensorExecutionOptions:get_CopyReporter():Lokad.Onnx.ICopyAccountant:this]
       48898540FDFFFF       mov      gword ptr [rbp-0x2C0], rax
       488B9540FDFFFF       mov      rdx, gword ptr [rbp-0x2C0]
       488B7DD0             mov      rdi, gword ptr [rbp-0x30]
       48BE18F64061947C0000 mov      rsi, 0x7C946140F618
       FF15896F0900         call     [Lokad.Onnx.Tensor`1[float]:RequireBatchOperand(Lokad.Onnx.Tensor`1[float],System.String,Lokad.Onnx.ICopyAccountant):Lokad.Onnx.Tensor`1[float]]
       488945D0             mov      gword ptr [rbp-0x30], rax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       3800                 cmp      byte  ptr [rax], al
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488D7848             lea      rdi, bword ptr [rax+0x48]
       FF15B95A0900         call     [Lokad.Onnx.TensorExecutionOptions:get_CopyReporter():Lokad.Onnx.ICopyAccountant:this]
       48898538FDFFFF       mov      gword ptr [rbp-0x2C8], rax
       488B9538FDFFFF       mov      rdx, gword ptr [rbp-0x2C8]
       488B7DC8             mov      rdi, gword ptr [rbp-0x38]
       48BE38F64061947C0000 mov      rsi, 0x7C946140F638
       FF154F6F0900         call     [Lokad.Onnx.Tensor`1[float]:RequireBatchOperand(Lokad.Onnx.Tensor`1[float],System.String,Lokad.Onnx.ICopyAccountant):Lokad.Onnx.Tensor`1[float]]
       488945C8             mov      gword ptr [rbp-0x38], rax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       3800                 cmp      byte  ptr [rax], al
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488D7848             lea      rdi, bword ptr [rax+0x48]
       FF157F5A0900         call     [Lokad.Onnx.TensorExecutionOptions:get_CopyReporter():Lokad.Onnx.ICopyAccountant:this]
       48898530FDFFFF       mov      gword ptr [rbp-0x2D0], rax
       488B9530FDFFFF       mov      rdx, gword ptr [rbp-0x2D0]
 
G_M000_IG03:                ;; offset=0x013F
       488B7DC0             mov      rdi, gword ptr [rbp-0x40]
       48BE58F64061947C0000 mov      rsi, 0x7C946140F658
       FF15FD590900         call     [Lokad.Onnx.Tensor`1[float]:RequireContiguous[float](Lokad.Onnx.Tensor`1[float],System.String,Lokad.Onnx.ICopyAccountant):Lokad.Onnx.DenseTensor`1[float]]
       488945C0             mov      gword ptr [rbp-0x40], rax
       488B7DD0             mov      rdi, gword ptr [rbp-0x30]
       393F                 cmp      dword ptr [rdi], edi
       FF150DF9F5FF         call     [Lokad.Onnx.Tensor`1[float]:get_Dimensions():System.ReadOnlySpan`1[int]:this]
       488985B0FEFFFF       mov      bword ptr [rbp-0x150], rax
       488995B8FEFFFF       mov      qword ptr [rbp-0x148], rdx
 
G_M000_IG04:                ;; offset=0x0171
       62F17E086F45EB       vmovdqu32 xmm0, xmmword ptr [rbp-0x150]
       C5FA7F8528FFFFFF     vmovdqu  xmmword ptr [rbp-0xD8], xmm0
 
G_M000_IG05:                ;; offset=0x0180
       488D8528FFFFFF       lea      rax, bword ptr [rbp-0xD8]
       48898538FFFFFF       mov      bword ptr [rbp-0xC8], rax
       488B8538FFFFFF       mov      rax, bword ptr [rbp-0xC8]
       8B4008               mov      eax, dword ptr [rax+0x08]
       8D50FE               lea      edx, [rax-0x02]
       488BBD38FFFFFF       mov      rdi, bword ptr [rbp-0xC8]
       33F6                 xor      esi, esi
       FF15465F0900         call     [System.ReadOnlySpan`1[int]:Slice(int,int):System.ReadOnlySpan`1[int]:this]
       488985A0FEFFFF       mov      bword ptr [rbp-0x160], rax
       488995A8FEFFFF       mov      qword ptr [rbp-0x158], rdx
 
G_M000_IG06:                ;; offset=0x01B8
       62F17E086F45EA       vmovdqu32 xmm0, xmmword ptr [rbp-0x160]
       C5FA7F8518FFFFFF     vmovdqu  xmmword ptr [rbp-0xE8], xmm0
 
G_M000_IG07:                ;; offset=0x01C7
       488DBD18FFFFFF       lea      rdi, [rbp-0xE8]
       FF1584E0EAFF         call     [System.ReadOnlySpan`1[int]:ToArray():int[]:this]
       488945B0             mov      gword ptr [rbp-0x50], rax
       488B7DD0             mov      rdi, gword ptr [rbp-0x30]
       393F                 cmp      dword ptr [rdi], edi
       FF158CF8F5FF         call     [Lokad.Onnx.Tensor`1[float]:get_Dimensions():System.ReadOnlySpan`1[int]:this]
       48898590FEFFFF       mov      bword ptr [rbp-0x170], rax
       48899598FEFFFF       mov      qword ptr [rbp-0x168], rdx
 
G_M000_IG08:                ;; offset=0x01F2
       62F17E086F45E9       vmovdqu32 xmm0, xmmword ptr [rbp-0x170]
       C5FA7F8528FFFFFF     vmovdqu  xmmword ptr [rbp-0xD8], xmm0
 
G_M000_IG09:                ;; offset=0x0201
       8B8530FFFFFF         mov      eax, dword ptr [rbp-0xD0]
       83C0FE               add      eax, -2
       89858CFEFFFF         mov      dword ptr [rbp-0x174], eax
       8B8530FFFFFF         mov      eax, dword ptr [rbp-0xD0]
       39858CFEFFFF         cmp      dword ptr [rbp-0x174], eax
       0F83460B0000         jae      G_M000_IG57
       8B858CFEFFFF         mov      eax, dword ptr [rbp-0x174]
       488B8D28FFFFFF       mov      rcx, bword ptr [rbp-0xD8]
       8B0481               mov      eax, dword ptr [rcx+4*rax]
       488B4DB8             mov      rcx, gword ptr [rbp-0x48]
       894138               mov      dword ptr [rcx+0x38], eax
       488B7DD0             mov      rdi, gword ptr [rbp-0x30]
       393F                 cmp      dword ptr [rdi], edi
       FF152BF8F5FF         call     [Lokad.Onnx.Tensor`1[float]:get_Dimensions():System.ReadOnlySpan`1[int]:this]
       48898578FEFFFF       mov      bword ptr [rbp-0x188], rax
       48899580FEFFFF       mov      qword ptr [rbp-0x180], rdx
 
G_M000_IG10:                ;; offset=0x0253
       C5FA6F8578FEFFFF     vmovdqu  xmm0, xmmword ptr [rbp-0x188]
       C5FA7F8528FFFFFF     vmovdqu  xmmword ptr [rbp-0xD8], xmm0
 
G_M000_IG11:                ;; offset=0x0263
       8B8530FFFFFF         mov      eax, dword ptr [rbp-0xD0]
       FFC8                 dec      eax
       898574FEFFFF         mov      dword ptr [rbp-0x18C], eax
       8B8530FFFFFF         mov      eax, dword ptr [rbp-0xD0]
       398574FEFFFF         cmp      dword ptr [rbp-0x18C], eax
       0F83E50A0000         jae      G_M000_IG57
       8B8574FEFFFF         mov      eax, dword ptr [rbp-0x18C]
       488B8D28FFFFFF       mov      rcx, bword ptr [rbp-0xD8]
       8B0481               mov      eax, dword ptr [rcx+4*rax]
       488B4DB8             mov      rcx, gword ptr [rbp-0x48]
       89413C               mov      dword ptr [rcx+0x3C], eax
       488B7DC8             mov      rdi, gword ptr [rbp-0x38]
       393F                 cmp      dword ptr [rdi], edi
       FF15CAF7F5FF         call     [Lokad.Onnx.Tensor`1[float]:get_Dimensions():System.ReadOnlySpan`1[int]:this]
       48898560FEFFFF       mov      bword ptr [rbp-0x1A0], rax
       48899568FEFFFF       mov      qword ptr [rbp-0x198], rdx
 
G_M000_IG12:                ;; offset=0x02B4
       62F17E086F45E6       vmovdqu32 xmm0, xmmword ptr [rbp-0x1A0]
       C5FA7F8528FFFFFF     vmovdqu  xmmword ptr [rbp-0xD8], xmm0
 
G_M000_IG13:                ;; offset=0x02C3
       8B8530FFFFFF         mov      eax, dword ptr [rbp-0xD0]
       FFC8                 dec      eax
       89855CFEFFFF         mov      dword ptr [rbp-0x1A4], eax
       8B8530FFFFFF         mov      eax, dword ptr [rbp-0xD0]
       39855CFEFFFF         cmp      dword ptr [rbp-0x1A4], eax
       0F83850A0000         jae      G_M000_IG57
       8B855CFEFFFF         mov      eax, dword ptr [rbp-0x1A4]
       488B8D28FFFFFF       mov      rcx, bword ptr [rbp-0xD8]
       8B0481               mov      eax, dword ptr [rcx+4*rax]
       488B4DB8             mov      rcx, gword ptr [rbp-0x48]
       894140               mov      dword ptr [rcx+0x40], eax
       488B7DB0             mov      rdi, gword ptr [rbp-0x50]
       FF15DC67D8FE         call     [System.ReadOnlySpan`1[int]:op_Implicit(int[]):System.ReadOnlySpan`1[int]]
       48898548FEFFFF       mov      bword ptr [rbp-0x1B8], rax
       48899550FEFFFF       mov      qword ptr [rbp-0x1B0], rdx
       488B7DD0             mov      rdi, gword ptr [rbp-0x30]
       393F                 cmp      dword ptr [rdi], edi
       FF1552F7F5FF         call     [Lokad.Onnx.Tensor`1[float]:get_Dimensions():System.ReadOnlySpan`1[int]:this]
       48898538FEFFFF       mov      bword ptr [rbp-0x1C8], rax
       48899540FEFFFF       mov      qword ptr [rbp-0x1C0], rdx
       488B7DD0             mov      rdi, gword ptr [rbp-0x30]
       FF154A6D0900         call     [Lokad.Onnx.Tensor`1[float]:BatchStrides(Lokad.Onnx.Tensor`1[float]):int[]]
       48898528FDFFFF       mov      gword ptr [rbp-0x2D8], rax
       4C8B8528FDFFFF       mov      r8, gword ptr [rbp-0x2D8]
       488BBD48FEFFFF       mov      rdi, bword ptr [rbp-0x1B8]
       488BB550FEFFFF       mov      rsi, qword ptr [rbp-0x1B0]
       488B9538FEFFFF       mov      rdx, bword ptr [rbp-0x1C8]
       488B8D40FEFFFF       mov      rcx, qword ptr [rbp-0x1C0]
       FF15326D0900         call     [Lokad.Onnx.Tensor`1[float]:BatchSteps(System.ReadOnlySpan`1[int],System.ReadOnlySpan`1[int],int[]):int[]]
       488945A8             mov      gword ptr [rbp-0x58], rax
       488B7DB0             mov      rdi, gword ptr [rbp-0x50]
       FF156C67D8FE         call     [System.ReadOnlySpan`1[int]:op_Implicit(int[]):System.ReadOnlySpan`1[int]]
       48898528FEFFFF       mov      bword ptr [rbp-0x1D8], rax
       48899530FEFFFF       mov      qword ptr [rbp-0x1D0], rdx
       488B7DC8             mov      rdi, gword ptr [rbp-0x38]
       393F                 cmp      dword ptr [rdi], edi
       FF15E2F6F5FF         call     [Lokad.Onnx.Tensor`1[float]:get_Dimensions():System.ReadOnlySpan`1[int]:this]
       48898518FEFFFF       mov      bword ptr [rbp-0x1E8], rax
       48899520FEFFFF       mov      qword ptr [rbp-0x1E0], rdx
       488B7DC8             mov      rdi, gword ptr [rbp-0x38]
       FF15DA6C0900         call     [Lokad.Onnx.Tensor`1[float]:BatchStrides(Lokad.Onnx.Tensor`1[float]):int[]]
       48898520FDFFFF       mov      gword ptr [rbp-0x2E0], rax
       4C8B8520FDFFFF       mov      r8, gword ptr [rbp-0x2E0]
       488BBD28FEFFFF       mov      rdi, bword ptr [rbp-0x1D8]
       488BB530FEFFFF       mov      rsi, qword ptr [rbp-0x1D0]
       488B9518FEFFFF       mov      rdx, bword ptr [rbp-0x1E8]
       488B8D20FEFFFF       mov      rcx, qword ptr [rbp-0x1E0]
       FF15C26C0900         call     [Lokad.Onnx.Tensor`1[float]:BatchSteps(System.ReadOnlySpan`1[int],System.ReadOnlySpan`1[int],int[]):int[]]
       488945A0             mov      gword ptr [rbp-0x60], rax
       488B7DB0             mov      rdi, gword ptr [rbp-0x50]
       FF15FC66D8FE         call     [System.ReadOnlySpan`1[int]:op_Implicit(int[]):System.ReadOnlySpan`1[int]]
       48898508FEFFFF       mov      bword ptr [rbp-0x1F8], rax
       48899510FEFFFF       mov      qword ptr [rbp-0x1F0], rdx
       488B7DC0             mov      rdi, gword ptr [rbp-0x40]
       393F                 cmp      dword ptr [rdi], edi
       FF1572F6F5FF         call     [Lokad.Onnx.Tensor`1[float]:get_Dimensions():System.ReadOnlySpan`1[int]:this]
       488985F8FDFFFF       mov      bword ptr [rbp-0x208], rax
       48899500FEFFFF       mov      qword ptr [rbp-0x200], rdx
 
G_M000_IG14:                ;; offset=0x040C
       488B45C0             mov      rax, gword ptr [rbp-0x40]
       4C8B4018             mov      r8, gword ptr [rax+0x18]
       488BBD08FEFFFF       mov      rdi, bword ptr [rbp-0x1F8]
       488BB510FEFFFF       mov      rsi, qword ptr [rbp-0x1F0]
       488B95F8FDFFFF       mov      rdx, bword ptr [rbp-0x208]
       488B8D00FEFFFF       mov      rcx, qword ptr [rbp-0x200]
       FF15626C0900         call     [Lokad.Onnx.Tensor`1[float]:BatchSteps(System.ReadOnlySpan`1[int],System.ReadOnlySpan`1[int],int[]):int[]]
       48894598             mov      gword ptr [rbp-0x68], rax
       488B7DB0             mov      rdi, gword ptr [rbp-0x50]
       FF159C66D8FE         call     [System.ReadOnlySpan`1[int]:op_Implicit(int[]):System.ReadOnlySpan`1[int]]
       488985E8FDFFFF       mov      bword ptr [rbp-0x218], rax
       488995F0FDFFFF       mov      qword ptr [rbp-0x210], rdx
       488BBDE8FDFFFF       mov      rdi, bword ptr [rbp-0x218]
       488BB5F0FDFFFF       mov      rsi, qword ptr [rbp-0x210]
       FF154A6C0900         call     [Lokad.Onnx.Tensor`1[float]:BatchCount(System.ReadOnlySpan`1[int]):int]
       894594               mov      dword ptr [rbp-0x6C], eax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       3800                 cmp      byte  ptr [rax], al
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488D7848             lea      rdi, bword ptr [rax+0x48]
       FF15FB4CF6FF         call     [Lokad.Onnx.TensorExecutionOptions:get_MaxDegreeOfParallelism():int:this]
       83F802               cmp      eax, 2
       7C55                 jl       SHORT G_M000_IG16
       837D9402             cmp      dword ptr [rbp-0x6C], 2
       7C40                 jl       SHORT G_M000_IG15
       48BF982E5F6B947C0000 mov      rdi, 0x7C946B5F2E98
       E8697A1E7D           call     CORINFO_HELP_COUNTPROFILE32
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       3800                 cmp      byte  ptr [rax], al
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488D7848             lea      rdi, bword ptr [rax+0x48]
       FF15CD4CF6FF         call     [Lokad.Onnx.TensorExecutionOptions:get_MaxDegreeOfParallelism():int:this]
       89851CFDFFFF         mov      dword ptr [rbp-0x2E4], eax
       8BBD1CFDFFFF         mov      edi, dword ptr [rbp-0x2E4]
       8B7594               mov      esi, dword ptr [rbp-0x6C]
       FF15E078DEFF         call     [System.Math:Min(int,int):int]
       8985E4FDFFFF         mov      dword ptr [rbp-0x21C], eax
       EB19                 jmp      SHORT G_M000_IG17
 
G_M000_IG15:                ;; offset=0x04C8
       48BF9C2E5F6B947C0000 mov      rdi, 0x7C946B5F2E9C
       E8297A1E7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG16:                ;; offset=0x04D7
       C785E4FDFFFF01000000 mov      dword ptr [rbp-0x21C], 1
 
G_M000_IG17:                ;; offset=0x04E1
       8B85E4FDFFFF         mov      eax, dword ptr [rbp-0x21C]
       894590               mov      dword ptr [rbp-0x70], eax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       4883C048             add      rax, 72
       488D3C24             lea      rdi, [rsp]
       488BF0               mov      rsi, rax
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
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       8B7038               mov      esi, dword ptr [rax+0x38]
       488B7DC8             mov      rdi, gword ptr [rbp-0x38]
       FF156D6B0900         call     [Lokad.Onnx.Tensor`1[float]:ResolvePackedKernel(Lokad.Onnx.TensorExecutionOptions,Lokad.Onnx.Tensor`1[float],int):Lokad.Onnx.DenseTensor`1[float]]
       48894588             mov      gword ptr [rbp-0x78], rax
       48837D8800           cmp      gword ptr [rbp-0x78], 0
       0F848D010000         je       G_M000_IG24
       837D9401             cmp      dword ptr [rbp-0x6C], 1
       0F840D010000         je       G_M000_IG21
       48BF302B5F6B947C0000 mov      rdi, 0x7C946B5F2B30
       FF15ECE2D7FE         call     [CORINFO_HELP_GET_GCSTATIC_BASE]
       48B8882D80818C7C0000 mov      rax, 0x7C8C81802D88
       488B00               mov      rax, gword ptr [rax]
       48898568FDFFFF       mov      gword ptr [rbp-0x298], rax
       488B45A0             mov      rax, gword ptr [rbp-0x60]
       48898560FDFFFF       mov      gword ptr [rbp-0x2A0], rax
       488B8568FDFFFF       mov      rax, gword ptr [rbp-0x298]
       48898558FDFFFF       mov      gword ptr [rbp-0x2A8], rax
       4883BD68FDFFFF00     cmp      gword ptr [rbp-0x298], 0
       0F859B000000         jne      G_M000_IG19
       48BFA02E5F6B947C0000 mov      rdi, 0x7C946B5F2EA0
       E832791E7D           call     CORINFO_HELP_COUNTPROFILE32
       48BFE82F386B947C0000 mov      rdi, 0x7C946B382FE8
       E8C334337D           call     CORINFO_HELP_NEWSFAST
       48898550FDFFFF       mov      gword ptr [rbp-0x2B0], rax
       48BF302B5F6B947C0000 mov      rdi, 0x7C946B5F2B30
       FF157CE2D7FE         call     [CORINFO_HELP_GET_GCSTATIC_BASE]
       48B8702B80818C7C0000 mov      rax, 0x7C8C81802B70
       488B00               mov      rax, gword ptr [rax]
 
G_M000_IG18:                ;; offset=0x0601
       48898510FDFFFF       mov      gword ptr [rbp-0x2F0], rax
       488BB510FDFFFF       mov      rsi, gword ptr [rbp-0x2F0]
       488BBD50FDFFFF       mov      rdi, gword ptr [rbp-0x2B0]
       48BAC09E556B947C0000 mov      rdx, 0x7C946B559EC0
       FF1542F9D7FE         call     [System.MulticastDelegate:CtorClosed(System.Object,nint):this]
       48BF302B5F6B947C0000 mov      rdi, 0x7C946B5F2B30
       FF153AE2D7FE         call     [CORINFO_HELP_GET_GCSTATIC_BASE]
       488BB550FDFFFF       mov      rsi, gword ptr [rbp-0x2B0]
       48BF882D80818C7C0000 mov      rdi, 0x7C8C81802D88
       E8548BD7FD           call     CORINFO_HELP_ASSIGN_REF
       488B8550FDFFFF       mov      rax, gword ptr [rbp-0x2B0]
       48898558FDFFFF       mov      gword ptr [rbp-0x2A8], rax
 
G_M000_IG19:                ;; offset=0x065A
       488BBD60FDFFFF       mov      rdi, gword ptr [rbp-0x2A0]
       488BB558FDFFFF       mov      rsi, gword ptr [rbp-0x2A8]
       FF156AD1F5FF         call     [System.Linq.Enumerable:All[int](System.Collections.Generic.IEnumerable`1[int],System.Func`2[int,bool]):bool]
       85C0                 test     eax, eax
       7476                 je       SHORT G_M000_IG23
 
G_M000_IG20:                ;; offset=0x0672
       48BFA42E5F6B947C0000 mov      rdi, 0x7C946B5F2EA4
       E87F781E7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG21:                ;; offset=0x0681
       48BFA82E5F6B947C0000 mov      rdi, 0x7C946B5F2EA8
       E870781E7D           call     CORINFO_HELP_COUNTPROFILE32
       8B4590               mov      eax, dword ptr [rbp-0x70]
       890424               mov      dword ptr [rsp], eax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       8B4038               mov      eax, dword ptr [rax+0x38]
       89442408             mov      dword ptr [rsp+0x08], eax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       8B403C               mov      eax, dword ptr [rax+0x3C]
       89442410             mov      dword ptr [rsp+0x10], eax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       8B4040               mov      eax, dword ptr [rax+0x40]
       89442418             mov      dword ptr [rsp+0x18], eax
       488B4588             mov      rax, gword ptr [rbp-0x78]
       4889442420           mov      gword ptr [rsp+0x20], rax
       488B7DD0             mov      rdi, gword ptr [rbp-0x30]
       488B75C0             mov      rsi, gword ptr [rbp-0x40]
       488B55B0             mov      rdx, gword ptr [rbp-0x50]
       488B4DA8             mov      rcx, gword ptr [rbp-0x58]
       4C8B4598             mov      r8, gword ptr [rbp-0x68]
       448B4D94             mov      r9d, dword ptr [rbp-0x6C]
       FF15026A0900         call     [Lokad.Onnx.Tensor`1[float]:RunPackedBatches(Lokad.Onnx.Tensor`1[float],Lokad.Onnx.Tensor`1[float],int[],int[],int[],int,int,int,int,int,Lokad.Onnx.DenseTensor`1[float])]
       90                   nop      
 
G_M000_IG22:                ;; offset=0x06DF
       4881C420030000       add      rsp, 800
       5D                   pop      rbp
       C3                   ret      
 
G_M000_IG23:                ;; offset=0x06E8
       48BFAC2E5F6B947C0000 mov      rdi, 0x7C946B5F2EAC
       E809781E7D           call     CORINFO_HELP_COUNTPROFILE32
 
G_M000_IG24:                ;; offset=0x06F7
       488B7DD0             mov      rdi, gword ptr [rbp-0x30]
       393F                 cmp      dword ptr [rdi], edi
       FF1595660900         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       488985D0FDFFFF       mov      gword ptr [rbp-0x230], rax
       488995D8FDFFFF       mov      qword ptr [rbp-0x228], rdx
 
G_M000_IG25:                ;; offset=0x0711
       62F17E086F45DD       vmovdqu32 xmm0, xmmword ptr [rbp-0x230]
       C5FA7F8508FFFFFF     vmovdqu  xmmword ptr [rbp-0xF8], xmm0
 
G_M000_IG26:                ;; offset=0x0720
       488DB570FFFFFF       lea      rsi, [rbp-0x90]
       488DBD08FFFFFF       lea      rdi, [rbp-0xF8]
       FF15BCEEF5FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG27:                ;; offset=0x0735
       488B7DC8             mov      rdi, gword ptr [rbp-0x38]
       393F                 cmp      dword ptr [rdi], edi
       FF1557660900         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       488985C0FDFFFF       mov      gword ptr [rbp-0x240], rax
       488995C8FDFFFF       mov      qword ptr [rbp-0x238], rdx
 
G_M000_IG28:                ;; offset=0x074F
       62F17E086F45DC       vmovdqu32 xmm0, xmmword ptr [rbp-0x240]
       C5FA7F8508FFFFFF     vmovdqu  xmmword ptr [rbp-0xF8], xmm0
 
G_M000_IG29:                ;; offset=0x075E
       488DB558FFFFFF       lea      rsi, [rbp-0xA8]
       488DBD08FFFFFF       lea      rdi, [rbp-0xF8]
       FF157EEEF5FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG30:                ;; offset=0x0773
       488B7DC0             mov      rdi, gword ptr [rbp-0x40]
       393F                 cmp      dword ptr [rdi], edi
       FF1519660900         call     [Lokad.Onnx.Tensor`1[float]:get_Storage():System.Memory`1[float]:this]
       488985B0FDFFFF       mov      gword ptr [rbp-0x250], rax
       488995B8FDFFFF       mov      qword ptr [rbp-0x248], rdx
 
G_M000_IG31:                ;; offset=0x078D
       62F17E086F45DB       vmovdqu32 xmm0, xmmword ptr [rbp-0x250]
       C5FA7F8508FFFFFF     vmovdqu  xmmword ptr [rbp-0xF8], xmm0
 
G_M000_IG32:                ;; offset=0x079C
       488DB540FFFFFF       lea      rsi, [rbp-0xC0]
       488DBD08FFFFFF       lea      rdi, [rbp-0xF8]
       FF1540EEF5FF         call     [System.Memory`1[float]:Pin():System.Buffers.MemoryHandle:this]
       90                   nop      
 
G_M000_IG33:                ;; offset=0x07B1
       488DBD70FFFFFF       lea      rdi, [rbp-0x90]
       FF154AEEF5FF         call     [System.Buffers.MemoryHandle:get_Pointer():ptr:this]
       488B4DB8             mov      rcx, gword ptr [rbp-0x48]
       48894120             mov      qword ptr [rcx+0x20], rax
       488DBD58FFFFFF       lea      rdi, [rbp-0xA8]
       FF1535EEF5FF         call     [System.Buffers.MemoryHandle:get_Pointer():ptr:this]
       488B4DB8             mov      rcx, gword ptr [rbp-0x48]
       48894128             mov      qword ptr [rcx+0x28], rax
       488DBD40FFFFFF       lea      rdi, [rbp-0xC0]
       FF1520EEF5FF         call     [System.Buffers.MemoryHandle:get_Pointer():ptr:this]
       488B4DB8             mov      rcx, gword ptr [rbp-0x48]
       48894130             mov      qword ptr [rcx+0x30], rax
       837D9001             cmp      dword ptr [rbp-0x70], 1
       0F8E4D010000         jle      G_M000_IG35
       48637594             movsxd   rsi, dword ptr [rbp-0x6C]
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E8A333337D           call     CORINFO_HELP_NEWARR_1_VC
       488B4DB8             mov      rcx, gword ptr [rbp-0x48]
       488D7908             lea      rdi, bword ptr [rcx+0x08]
       488BF0               mov      rsi, rax
       E88389D7FD           call     CORINFO_HELP_ASSIGN_REF
       48637594             movsxd   rsi, dword ptr [rbp-0x6C]
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E88033337D           call     CORINFO_HELP_NEWARR_1_VC
       488B4DB8             mov      rcx, gword ptr [rbp-0x48]
       488D7910             lea      rdi, bword ptr [rcx+0x10]
       488BF0               mov      rsi, rax
       E86089D7FD           call     CORINFO_HELP_ASSIGN_REF
       48637594             movsxd   rsi, dword ptr [rbp-0x6C]
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E85D33337D           call     CORINFO_HELP_NEWARR_1_VC
       488B4DB8             mov      rcx, gword ptr [rbp-0x48]
       488D7918             lea      rdi, bword ptr [rcx+0x18]
       488BF0               mov      rsi, rax
       E83D89D7FD           call     CORINFO_HELP_ASSIGN_REF
       488B7DB0             mov      rdi, gword ptr [rbp-0x50]
       FF157362D8FE         call     [System.ReadOnlySpan`1[int]:op_Implicit(int[]):System.ReadOnlySpan`1[int]]
       48898598FDFFFF       mov      bword ptr [rbp-0x268], rax
       488995A0FDFFFF       mov      qword ptr [rbp-0x260], rdx
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488B4010             mov      rax, gword ptr [rax+0x10]
       48890424             mov      gword ptr [rsp], rax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488B4018             mov      rax, gword ptr [rax+0x18]
       4889442408           mov      gword ptr [rsp+0x08], rax
       488BBD98FDFFFF       mov      rdi, bword ptr [rbp-0x268]
       488BB5A0FDFFFF       mov      rsi, qword ptr [rbp-0x260]
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       4C8B4808             mov      r9, gword ptr [rax+0x08]
       488B55A8             mov      rdx, gword ptr [rbp-0x58]
       488B4DA0             mov      rcx, gword ptr [rbp-0x60]
       4C8B4598             mov      r8, gword ptr [rbp-0x68]
       FF153C680900         call     [Lokad.Onnx.Tensor`1[float]:FillBatchOffsets(System.ReadOnlySpan`1[int],int[],int[],int[],int[],int[],int[])]
       48BF70105F6B947C0000 mov      rdi, 0x7C946B5F1070
       E8D531337D           call     CORINFO_HELP_NEWSFAST
       48898590FDFFFF       mov      gword ptr [rbp-0x270], rax
       488BBD90FDFFFF       mov      rdi, gword ptr [rbp-0x270]
 
G_M000_IG34:                ;; offset=0x08D9
       FF1531680900         call     [System.Threading.Tasks.ParallelOptions:.ctor():this]
       488BBD90FDFFFF       mov      rdi, gword ptr [rbp-0x270]
       8B7590               mov      esi, dword ptr [rbp-0x70]
       393F                 cmp      dword ptr [rdi], edi
       FF1537680900         call     [System.Threading.Tasks.ParallelOptions:set_MaxDegreeOfParallelism(int):this]
       48BF98115F6B947C0000 mov      rdi, 0x7C946B5F1198
       E8A031337D           call     CORINFO_HELP_NEWSFAST
       48898588FDFFFF       mov      gword ptr [rbp-0x278], rax
       488BBD88FDFFFF       mov      rdi, gword ptr [rbp-0x278]
       488B75B8             mov      rsi, gword ptr [rbp-0x48]
       48BAD89E556B947C0000 mov      rdx, 0x7C946B559ED8
       FF1546F6D7FE         call     [System.MulticastDelegate:CtorClosed(System.Object,nint):this]
       488DBD70FDFFFF       lea      rdi, [rbp-0x290]
       4C8B8588FDFFFF       mov      r8, gword ptr [rbp-0x278]
       8B5594               mov      edx, dword ptr [rbp-0x6C]
       488B8D90FDFFFF       mov      rcx, gword ptr [rbp-0x270]
       33F6                 xor      esi, esi
       FF15FE670900         call     [System.Threading.Tasks.Parallel:For(int,int,System.Threading.Tasks.ParallelOptions,System.Action`1[int]):System.Threading.Tasks.ParallelLoopResult]
       E9E7030000           jmp      G_M000_IG50
 
G_M000_IG35:                ;; offset=0x0947
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488B4020             mov      rax, qword ptr [rax+0x20]
       48898500FFFFFF       mov      qword ptr [rbp-0x100], rax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488B4028             mov      rax, qword ptr [rax+0x28]
       488985F8FEFFFF       mov      qword ptr [rbp-0x108], rax
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       488B4030             mov      rax, qword ptr [rax+0x30]
       488985F0FEFFFF       mov      qword ptr [rbp-0x110], rax
       488B45B0             mov      rax, gword ptr [rbp-0x50]
       8B4008               mov      eax, dword ptr [rax+0x08]
       8985ECFEFFFF         mov      dword ptr [rbp-0x114], eax
       4863B5ECFEFFFF       movsxd   rsi, dword ptr [rbp-0x114]
       48BF58B1276A947C0000 mov      rdi, 0x7C946A27B158
       E81932337D           call     CORINFO_HELP_NEWARR_1_VC
       488985E0FEFFFF       mov      gword ptr [rbp-0x120], rax
       33C0                 xor      eax, eax
       8985DCFEFFFF         mov      dword ptr [rbp-0x124], eax
       33C0                 xor      eax, eax
       8985D8FEFFFF         mov      dword ptr [rbp-0x128], eax
       33C0                 xor      eax, eax
       8985D4FEFFFF         mov      dword ptr [rbp-0x12C], eax
       33C0                 xor      eax, eax
       8985D0FEFFFF         mov      dword ptr [rbp-0x130], eax
       E91D030000           jmp      G_M000_IG46
 
G_M000_IG36:                ;; offset=0x09C3
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       4883C048             add      rax, 72
       488D3C24             lea      rdi, [rsp]
       488BF0               mov      rsi, rax
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
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       8B7838               mov      edi, dword ptr [rax+0x38]
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       8B703C               mov      esi, dword ptr [rax+0x3C]
       488B45B8             mov      rax, gword ptr [rbp-0x48]
       8B5040               mov      edx, dword ptr [rax+0x40]
       488B8500FFFFFF       mov      rax, qword ptr [rbp-0x100]
       48638DDCFEFFFF       movsxd   rcx, dword ptr [rbp-0x124]
       488D0C88             lea      rcx, [rax+4*rcx]
       488B85F8FEFFFF       mov      rax, qword ptr [rbp-0x108]
       4C6385D8FEFFFF       movsxd   r8, dword ptr [rbp-0x128]
       4E8D0480             lea      r8, [rax+4*r8]
       488B85F0FEFFFF       mov      rax, qword ptr [rbp-0x110]
       4C638DD4FEFFFF       movsxd   r9, dword ptr [rbp-0x12C]
       4E8D0C88             lea      r9, [rax+4*r9]
       FF15E4660900         call     [Lokad.Onnx.Tensor`1[float]:RunIsolatedShortWideKernel(int,int,int,ptr,ptr,ptr,Lokad.Onnx.TensorExecutionOptions)]
       8B85ECFEFFFF         mov      eax, dword ptr [rbp-0x114]
       FFC8                 dec      eax
       8985CCFEFFFF         mov      dword ptr [rbp-0x134], eax
       E907020000           jmp      G_M000_IG42
 
G_M000_IG37:                ;; offset=0x0A87
       48BFB02E5F6B947C0000 mov      rdi, 0x7C946B5F2EB0
       E86A741E7D           call     CORINFO_HELP_COUNTPROFILE32
       E928020000           jmp      G_M000_IG45
 
G_M000_IG38:                ;; offset=0x0A9B
       E8309ED7FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG39:                ;; offset=0x0AA1
       488B85E0FEFFFF       mov      rax, gword ptr [rbp-0x120]
       8B8DCCFEFFFF         mov      ecx, dword ptr [rbp-0x134]
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       73E8                 jae      SHORT G_M000_IG38
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       488985A8FDFFFF       mov      bword ptr [rbp-0x258], rax
       488B85A8FDFFFF       mov      rax, bword ptr [rbp-0x258]
       FF00                 inc      dword ptr [rax]
       488B45A8             mov      rax, gword ptr [rbp-0x58]
       8B8DCCFEFFFF         mov      ecx, dword ptr [rbp-0x134]
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       73C2                 jae      SHORT G_M000_IG38
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       8B00                 mov      eax, dword ptr [rax]
       0385DCFEFFFF         add      eax, dword ptr [rbp-0x124]
       8985DCFEFFFF         mov      dword ptr [rbp-0x124], eax
       488B45A0             mov      rax, gword ptr [rbp-0x60]
       8B8DCCFEFFFF         mov      ecx, dword ptr [rbp-0x134]
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       739E                 jae      SHORT G_M000_IG38
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       8B00                 mov      eax, dword ptr [rax]
       0385D8FEFFFF         add      eax, dword ptr [rbp-0x128]
       8985D8FEFFFF         mov      dword ptr [rbp-0x128], eax
       488B4598             mov      rax, gword ptr [rbp-0x68]
       8B8DCCFEFFFF         mov      ecx, dword ptr [rbp-0x134]
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       0F8376FFFFFF         jae      G_M000_IG38
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       8B00                 mov      eax, dword ptr [rax]
       0385D4FEFFFF         add      eax, dword ptr [rbp-0x12C]
       8985D4FEFFFF         mov      dword ptr [rbp-0x12C], eax
       488B85E0FEFFFF       mov      rax, gword ptr [rbp-0x120]
       8B8DCCFEFFFF         mov      ecx, dword ptr [rbp-0x134]
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       0F834BFFFFFF         jae      G_M000_IG38
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       8B00                 mov      eax, dword ptr [rax]
       488B4DB0             mov      rcx, gword ptr [rbp-0x50]
       8B95CCFEFFFF         mov      edx, dword ptr [rbp-0x134]
       3B5108               cmp      edx, dword ptr [rcx+0x08]
       0F832FFFFFFF         jae      G_M000_IG38
       8BFA                 mov      edi, edx
       488D4CB910           lea      rcx, bword ptr [rcx+4*rdi+0x10]
       3B01                 cmp      eax, dword ptr [rcx]
       0F8C0CFFFFFF         jl       G_M000_IG37
 
G_M000_IG40:                ;; offset=0x0B7B
       48BFB42E5F6B947C0000 mov      rdi, 0x7C946B5F2EB4
       E876731E7D           call     CORINFO_HELP_COUNTPROFILE32
       488B85E0FEFFFF       mov      rax, gword ptr [rbp-0x120]
       8B8DCCFEFFFF         mov      ecx, dword ptr [rbp-0x134]
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       0F83FBFEFFFF         jae      G_M000_IG38
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       33C9                 xor      ecx, ecx
       8908                 mov      dword ptr [rax], ecx
       488B45A8             mov      rax, gword ptr [rbp-0x58]
       8B8DCCFEFFFF         mov      ecx, dword ptr [rbp-0x134]
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       0F83DDFEFFFF         jae      G_M000_IG38
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       8B00                 mov      eax, dword ptr [rax]
       488B4DB0             mov      rcx, gword ptr [rbp-0x50]
       8B95CCFEFFFF         mov      edx, dword ptr [rbp-0x134]
       3B5108               cmp      edx, dword ptr [rcx+0x08]
       0F83C1FEFFFF         jae      G_M000_IG38
       8BFA                 mov      edi, edx
       488D4CB910           lea      rcx, bword ptr [rcx+4*rdi+0x10]
       0FAF01               imul     eax, dword ptr [rcx]
       8B8DDCFEFFFF         mov      ecx, dword ptr [rbp-0x124]
       2BC8                 sub      ecx, eax
       898DDCFEFFFF         mov      dword ptr [rbp-0x124], ecx
       488B45A0             mov      rax, gword ptr [rbp-0x60]
       8B8DCCFEFFFF         mov      ecx, dword ptr [rbp-0x134]
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       0F8396FEFFFF         jae      G_M000_IG38
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       8B00                 mov      eax, dword ptr [rax]
       488B4DB0             mov      rcx, gword ptr [rbp-0x50]
       8B95CCFEFFFF         mov      edx, dword ptr [rbp-0x134]
       3B5108               cmp      edx, dword ptr [rcx+0x08]
       0F837AFEFFFF         jae      G_M000_IG38
       8BFA                 mov      edi, edx
       488D4CB910           lea      rcx, bword ptr [rcx+4*rdi+0x10]
       0FAF01               imul     eax, dword ptr [rcx]
       8B8DD8FEFFFF         mov      ecx, dword ptr [rbp-0x128]
       2BC8                 sub      ecx, eax
       898DD8FEFFFF         mov      dword ptr [rbp-0x128], ecx
       488B4598             mov      rax, gword ptr [rbp-0x68]
       8B8DCCFEFFFF         mov      ecx, dword ptr [rbp-0x134]
       3B4808               cmp      ecx, dword ptr [rax+0x08]
       0F834FFEFFFF         jae      G_M000_IG38
       8BD1                 mov      edx, ecx
       488D449010           lea      rax, bword ptr [rax+4*rdx+0x10]
       8B00                 mov      eax, dword ptr [rax]
       488B4DB0             mov      rcx, gword ptr [rbp-0x50]
       8B95CCFEFFFF         mov      edx, dword ptr [rbp-0x134]
 
G_M000_IG41:                ;; offset=0x0C5F
       3B5108               cmp      edx, dword ptr [rcx+0x08]
       0F8333FEFFFF         jae      G_M000_IG38
       8BFA                 mov      edi, edx
       488D4CB910           lea      rcx, bword ptr [rcx+4*rdi+0x10]
       0FAF01               imul     eax, dword ptr [rcx]
       8B8DD4FEFFFF         mov      ecx, dword ptr [rbp-0x12C]
       2BC8                 sub      ecx, eax
       898DD4FEFFFF         mov      dword ptr [rbp-0x12C], ecx
       8B85CCFEFFFF         mov      eax, dword ptr [rbp-0x134]
       FFC8                 dec      eax
       8985CCFEFFFF         mov      dword ptr [rbp-0x134], eax
 
G_M000_IG42:                ;; offset=0x0C8E
       8B8548FDFFFF         mov      eax, dword ptr [rbp-0x2B8]
       FFC8                 dec      eax
       898548FDFFFF         mov      dword ptr [rbp-0x2B8], eax
       83BD48FDFFFF00       cmp      dword ptr [rbp-0x2B8], 0
       7F11                 jg       SHORT G_M000_IG44
 
G_M000_IG43:                ;; offset=0x0CA5
       488DBD48FDFFFF       lea      rdi, [rbp-0x2B8]
       BE6C030000           mov      esi, 876
       E8CA2B337D           call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG44:                ;; offset=0x0CB6
       83BDCCFEFFFF00       cmp      dword ptr [rbp-0x134], 0
       0F8DDEFDFFFF         jge      G_M000_IG39
 
G_M000_IG45:                ;; offset=0x0CC3
       48BFB82E5F6B947C0000 mov      rdi, 0x7C946B5F2EB8
       E82E721E7D           call     CORINFO_HELP_COUNTPROFILE32
       8B85D0FEFFFF         mov      eax, dword ptr [rbp-0x130]
       FFC0                 inc      eax
       8985D0FEFFFF         mov      dword ptr [rbp-0x130], eax
 
G_M000_IG46:                ;; offset=0x0CE0
       8B8548FDFFFF         mov      eax, dword ptr [rbp-0x2B8]
       FFC8                 dec      eax
       898548FDFFFF         mov      dword ptr [rbp-0x2B8], eax
       83BD48FDFFFF00       cmp      dword ptr [rbp-0x2B8], 0
       7F11                 jg       SHORT G_M000_IG48
 
G_M000_IG47:                ;; offset=0x0CF7
       488DBD48FDFFFF       lea      rdi, [rbp-0x2B8]
       BE77030000           mov      esi, 887
       E8782B337D           call     CORINFO_HELP_PATCHPOINT
 
G_M000_IG48:                ;; offset=0x0D08
       8B85D0FEFFFF         mov      eax, dword ptr [rbp-0x130]
       3B4594               cmp      eax, dword ptr [rbp-0x6C]
       0F8CACFCFFFF         jl       G_M000_IG36
       48BFBC2E5F6B947C0000 mov      rdi, 0x7C946B5F2EBC
       E8DA711E7D           call     CORINFO_HELP_COUNTPROFILE32
       90                   nop      
 
G_M000_IG49:                ;; offset=0x0D27
       E842000000           call     G_M000_IG58
       EB0D                 jmp      SHORT G_M000_IG52
 
G_M000_IG50:                ;; offset=0x0D2E
       E83B000000           call     G_M000_IG58
       90                   nop      
 
G_M000_IG51:                ;; offset=0x0D34
       E85B000000           call     G_M000_IG61
       EB07                 jmp      SHORT G_M000_IG53
 
G_M000_IG52:                ;; offset=0x0D3B
       E854000000           call     G_M000_IG61
       EB07                 jmp      SHORT G_M000_IG54
 
G_M000_IG53:                ;; offset=0x0D42
       E873000000           call     G_M000_IG64
       EB06                 jmp      SHORT G_M000_IG55
 
G_M000_IG54:                ;; offset=0x0D49
       E86C000000           call     G_M000_IG64
       90                   nop      
 
G_M000_IG55:                ;; offset=0x0D4F
       48BFCC2E5F6B947C0000 mov      rdi, 0x7C946B5F2ECC
       E8A2711E7D           call     CORINFO_HELP_COUNTPROFILE32
       90                   nop      
 
G_M000_IG56:                ;; offset=0x0D5F
       4881C420030000       add      rsp, 800
       5D                   pop      rbp
       C3                   ret      
 
G_M000_IG57:                ;; offset=0x0D68
       E8639BD7FE           call     CORINFO_HELP_RNGCHKFAIL
       CC                   int3     
 
G_M000_IG58:                ;; offset=0x0D6E
       4883EC38             sub      rsp, 56
 
G_M000_IG59:                ;; offset=0x0D72
       48BFC02E5F6B947C0000 mov      rdi, 0x7C946B5F2EC0
       E87F711E7D           call     CORINFO_HELP_COUNTPROFILE32
       488DBD40FFFFFF       lea      rdi, [rbp-0xC0]
       FF150AE9F5FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG60:                ;; offset=0x0D8F
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG61:                ;; offset=0x0D94
       4883EC38             sub      rsp, 56
 
G_M000_IG62:                ;; offset=0x0D98
       48BFC42E5F6B947C0000 mov      rdi, 0x7C946B5F2EC4
       E859711E7D           call     CORINFO_HELP_COUNTPROFILE32
       488DBD58FFFFFF       lea      rdi, [rbp-0xA8]
       FF15E4E8F5FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG63:                ;; offset=0x0DB5
       4883C438             add      rsp, 56
       C3                   ret      
 
G_M000_IG64:                ;; offset=0x0DBA
       4883EC38             sub      rsp, 56
 
G_M000_IG65:                ;; offset=0x0DBE
       48BFC82E5F6B947C0000 mov      rdi, 0x7C946B5F2EC8
       E833711E7D           call     CORINFO_HELP_COUNTPROFILE32
       488DBD70FFFFFF       lea      rdi, [rbp-0x90]
       FF15BEE8F5FF         call     [System.Buffers.MemoryHandle:Dispose():this]
       90                   nop      
 
G_M000_IG66:                ;; offset=0x0DDB
       4883C438             add      rsp, 56
       C3                   ret      
 
; Total bytes of code 3552
