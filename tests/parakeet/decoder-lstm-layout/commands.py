"""One build, compiled comparison and three correctness modes; no timing score."""
DOTNET = '/home/vermorel/.dotnet/dotnet'
FLAGS = ['--tl:off', '--nologo', '-v', 'minimal', '-p:EnableSourceControlManagerQueries=false',
    '-p:EnableSourceLink=false', '-p:UseSharedCompilation=false', '-nr:false', '-p:NuGetAudit=false']


def command_for(base, name, spec):
    if name == 'sdk-version': return [DOTNET, '--version'], True, 2
    project = base/'source/tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'
    if name.startswith('backend-'):
        action = name.split('-')[1]
        command = [DOTNET, action, project, *FLAGS]
        command += ['--source', spec['feed'], '--packages', base/'packages'] if action == 'restore' else ['-c', 'Release', '--no-restore', '--disable-build-servers']
        return command, True, 2
    if name == 'inventory':
        return [DOTNET, base/'bridge/Bridge.dll', base/'runtimes/current',
                base/'runtimes/candidate', base/'inventory/instructions.json', base/'runtimes/current'], False, 2
    mode = name.removeprefix('contracts-'); assert mode in ['normal', 'noavx512', 'scalar']
    return [spec['python'], '-B', base/'tools/testmode.py', mode, base, base/name,
        DOTNET, 'test', project, '-c', 'Release', *FLAGS, '--no-build', '--no-restore',
        '--filter', 'FullyQualifiedName~Lstm|FullyQualifiedName~LSTM',
        '--logger', 'trx;LogFileName=contracts.trx', '--results-directory', base/name], True, 2
