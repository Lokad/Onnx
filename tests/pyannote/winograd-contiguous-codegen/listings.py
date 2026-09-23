"""Retain all raw JIT tiers and explicitly rejoin the driver's known stdout line."""
import hashlib
import re

MARKER='Diagnostic complete: 432 exact ordinary graph outputs.\n'

def listings(path):
    text=path.read_text(encoding='utf8')
    starts=list(re.finditer(r'; Assembly listing for method (.+) \(([^\n]+)\)\n',text))
    result=[]
    for index,start in enumerate(starts):
        end=starts[index+1].start() if index+1<len(starts) else len(text)
        raw=text[start.start():end]
        repairs=[dict(offset=m.start(),removed=MARKER) for m in re.finditer(re.escape(MARKER),raw)]
        body=raw.replace(MARKER,'');sizes=re.findall(r'; Total bytes of code (\d+)',body)
        blocks=list(re.finditer(r'^(G_M\d+_IG\d+):[^\n]*\n',body,re.M));reductions=[]
        labels=[b.group(1) for b in blocks]
        ordinals=[int(label.rsplit('IG',1)[1]) for label in labels]
        complete=(len(sizes)==1 and len(repairs)<=1 and 'Diagnostic' not in body
            and len(labels)>0 and len(labels)==len(set(labels))
            and ordinals==list(range(1,len(labels)+1))
            and set(re.findall(r'\bG_M\d+_IG\d+\b',body))==set(labels))
        for i,block in enumerate(blocks):
            part=body[block.start():blocks[i+1].start() if i+1<len(blocks) else len(body)]
            fmas=len(re.findall(r'\bvfmadd\d*ps\b',part))
            if fmas:
                stack=[line for line in part.splitlines() if re.search(r'\b[xyz]mm(?:word)?\b|\b[xyz]mm\d+',line)
                       and re.search(r'\[(?:rbp|rsp)(?:[+\-\]])',line)]
                reductions.append(dict(label=block.group(1),fma_instructions=fmas,
                    input_broadcasts=len(re.findall(r'\bvbroadcastss\b',part)),vector_stack_references=stack,
                    integer_multiplies=len(re.findall(r'\bimul\b',part)),
                    arithmetic_shifts=len(re.findall(r'\bsar\b',part)),
                    scalar_stack_references=[line for line in part.splitlines()
                        if re.search(r'\[(?:rbp|rsp)(?:[+\-\]])',line) and line not in stack],body=part))
        result.append(dict(method=start.group(1),tier=start.group(2),code_bytes=[int(x) for x in sizes],
            complete_body=complete,complete_uninterleaved=complete and not repairs,
            line=text.count('\n',0,start.start())+1,reductions=reductions,body=body,raw_body=raw,
            managed_stdout_repairs=repairs,raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            body_sha256=hashlib.sha256(body.encode()).hexdigest()))
    return result
