using System.Diagnostics.Tracing;
using System.Text.Json;
using Microsoft.Diagnostics.NETCore.Client;
using Microsoft.Diagnostics.Tracing;
using Microsoft.Diagnostics.Tracing.Parsers.Clr;

if(args.Length!=3)throw new ArgumentException("PID output-directory gate-file");
int pid=int.Parse(args[0]);string output=args[1],gate=args[2];
if(Directory.Exists(output)||File.Exists(gate))throw new IOException("Existing output or gate");
Directory.CreateDirectory(output);
using var events=new StreamWriter(new FileStream(Path.Combine(output,"events.jsonl"),FileMode.CreateNew));
var providers=new[]{new EventPipeProvider("Microsoft-Windows-DotNETRuntime",EventLevel.Verbose,0x1000000019),new EventPipeProvider("Lokad-Packed-Phase",EventLevel.Informational,-1)};
var client=new DiagnosticsClient(pid);
using var session=client.StartEventPipeSession(providers,false,64);
using var source=new EventPipeEventSource(session.EventStream);
int count=0,markers=0,methods=0;
void Record(TraceEvent e)
{
    var payload=new Dictionary<string,string?>();
    foreach(string name in e.PayloadNames)payload[name]=e.PayloadByName(name)?.ToString();
    string? tier=e is MethodLoadUnloadVerboseTraceData method?method.OptimizationTier.ToString():null;
    events.WriteLine(JsonSerializer.Serialize(new{provider=e.ProviderName,name=e.EventName,id=(int)e.ID,thread=e.ThreadID,pid=e.ProcessID,ms=e.TimeStampRelativeMSec,payload,tier}));
    count++;if(e.ProviderName=="Lokad-Packed-Phase")markers++;if(e.EventName.Contains("JittingStarted"))methods++;
}
source.Clr.All+=Record;
source.Dynamic.All+=e=>{if(e.ProviderName=="Lokad-Packed-Phase")Record(e);};
using(var ready=new FileStream(gate,FileMode.CreateNew)){ready.WriteByte(1);}
source.Process();events.Flush();
File.WriteAllText(Path.Combine(output,"result.json"),JsonSerializer.Serialize(new{pid,events=count,markers,methods,lost=source.EventsLost,completed=true,collector_pid=Environment.ProcessId},new JsonSerializerOptions{WriteIndented=true}));
if(source.EventsLost!=0||markers<2||methods<1)throw new InvalidDataException("Incomplete diagnostic events");
