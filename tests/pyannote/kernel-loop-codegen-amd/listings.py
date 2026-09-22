"""Retain every emitted JIT tier and its reduction blocks."""
import re


def listings(path):
    text=path.read_text(encoding='utf8')
    starts=list(re.finditer(r'; Assembly listing for method (.+) \(([^\n]+)\)\n',text))
    result=[]
    for index,start in enumerate(starts):
        end=starts[index+1].start() if index+1<len(starts) else len(text)
        body=text[start.start():end];sizes=re.findall(r'; Total bytes of code (\d+)',body)
        blocks=list(re.finditer(r'^(G_M\d+_IG\d+):[^\n]*\n',body,re.M));reductions=[]
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
            complete_uninterleaved=len(sizes)==1,line=text.count('\n',0,start.start())+1,reductions=reductions,body=body))
    return result

