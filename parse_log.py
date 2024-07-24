import Drain,DrainTB

input_dir = 'log_data/'  # The input directory of log file
output_dir = 'parse_result/'  # The output directory of parsing results
#log_file = 'openstack.log'  # The input log file name
log_file = "Spirit.log"
# log_file = "Thunderbird_mini.log"

hdfs_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
bgl_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'
openstack_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'
thunderbird_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
Spirit_format = '<Label> <TimeStamp> <Date> <User> <Month> <Day> <Time> <UserGroup> <Component>(\[<PID>\])?(:)? <Content>'
#thunderbird_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'
#
#thunderbird_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'
#'<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'

bgl_regex = [
    r'core\.\d+',
    r'(?:\/[\*\w\.-]+)+',  # path
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'0x[0-9a-f]+(?: [0-9a-f]{8})*',  # hex
    r'[0-9a-f]{8}(?: [0-9a-f]{8})*',
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
]
thunderbird_regex = [
        r'(0x)[0-9a-fA-F]+',  # hexadecimal
        r'\d+\.\d+\.\d+\.\d+',
        r'(?<=Warning: we failed to resolve data source name )[\w\s]+',
        r'\d+'
    ]
# thunderbird_regex =  [ r'(\d+\.){ 3 }\d+' ]
hdfs_regex = [
    r'blk_(|-)[0-9]+',  # block id
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
]

openstack_regex = [
    r'(?<=\[instance: ).*?(?=\])',
    r'(?<=\[req).*(?= -)',
    r'(?<=image ).*(?= at)',
    r'(?<=[^a-zA-Z0-9])(?:\/[\*\w\.-]+)+',  # path
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
    r'(?<=\s|=)\d+(?:\.\d+)?'
]

Spirit_regex = [r'0x.*?\s', r'(\d+\.){3}\d+(:\d+)?', r'#(\d+)#', r'<(\d{14})\.(.*?)\@', r'(\d+)-(\d+)', r'(\d+)kB', r'\d{2}:\d{2}:\d{2}']

st = 0.5  # Similarity threshold
depth = 4  # Depth of all leaf nodes
# depth = 5  # openstack

# parser = Drain.LogParser(thunderbird_format, indir=input_dir,
#                          outdir=output_dir, depth=3, st=0.3, rex=thunderbird_regex)
# # parser.parse(log_file)
# keep_para = False
parser = Drain.LogParser(Spirit_format, indir=input_dir, outdir=output_dir, depth=4, st=0.5,
                           rex=Spirit_regex)
# parser = Drain.LogParser(thunderbird_format, indir=input_dir, outdir=output_dir, depth=3, st=0.3,
#                            rex=thunderbird_regex, keep_para=keep_para, maxChild=1000)

# parser = Drain.LogParser(bgl_format, indir=input_dir,
#                          outdir=output_dir, depth=4, st=0.5, rex=bgl_regex)
parser.parse(log_file)