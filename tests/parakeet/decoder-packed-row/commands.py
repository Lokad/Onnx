"""Pure command construction, usable by an offline auditor on Windows."""
DOTNET = '/home/vermorel/.dotnet/dotnet'
FLAGS = ['--tl:off', '--nologo', '-v', 'minimal', '-p:EnableSourceControlManagerQueries=false',
    '-p:EnableSourceLink=false', '-p:UseSharedCompilation=false', '-nr:false', '-p:NuGetAudit=false']


def command_for(base, name, spec):
    if name == 'sdk-version': return [DOTNET, '--version'], True, 2
    if name.startswith(('core-', 'contracts-')):
        kind, action = name.split('-')
        project = base / ('source/src/Lokad.Onnx/Lokad.Onnx.csproj' if kind == 'core' else 'source/contracts/Contracts.csproj')
        command = [DOTNET, action, project, *FLAGS]
        command += ['--source', spec['feed'], '--packages', base/'packages'] if action == 'restore' else ['-c', 'Release', '--no-restore', '--disable-build-servers']
        return command, True, 2
    if name == 'inventory':
        return [DOTNET, base/'bridge/Bridge.dll', base/'runtimes/current', base/'runtimes/candidate',
            base/'inventory/instructions.json', base/'runtimes/current'], False, 2
    role, mode = name.split('-')
    assert role in ['current', 'candidate'] and mode in ['normal', 'noavx512', 'scalar']
    command = [DOTNET, base/'runtimes'/role/'Lokad.Onnx.Backend.Tests.dll', base, role, mode, base/name]
    if mode != 'normal': command = [spec['python'], '-B', base/'tools/execmode.py', mode, *command]
    return command, False, 2
