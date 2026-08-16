"""
generate_dataset.py
--------------------
Expanded training dataset: 100 samples per category (1000 total).

APPROACH (transparency for report/viva):
- 30 hand-written factual sentences per category (original, foundational)
- 70 additional sentences per category generated via TEMPLATE AUGMENTATION:
  combining pairs of category-specific technical terms with varied
  generic sentence templates. This is a standard, legitimate data
  augmentation technique for small text-classification datasets.
  Templates deliberately avoid stating the category name itself, so the
  classifier must learn from genuine technical vocabulary, not label leakage.
- random_state fixed (seed=42) for reproducibility.
"""
import csv
import random

random.seed(42)

# ============================================================
# 30 hand-written base sentences per category (original)
# ============================================================
base_data = {
"Programming": [
"A variable is a named storage location in memory used to hold data values during program execution.",
"Functions allow developers to encapsulate reusable blocks of code that can be called multiple times.",
"Object oriented programming relies on classes and objects to model real world entities using inheritance and polymorphism.",
"Loops such as for and while allow a program to repeat a block of statements until a condition is met.",
"Recursion is a technique where a function calls itself to solve smaller instances of the same problem.",
"Python is a high level interpreted programming language known for its readable syntax and dynamic typing.",
"Exception handling using try catch blocks allows programs to gracefully handle runtime errors.",
"An array is a collection of elements identified by index or key stored in contiguous memory locations.",
"Debugging is the process of finding and fixing errors or bugs within the source code of a program.",
"Version control systems like git allow multiple developers to track changes and collaborate on source code.",
"A compiler translates source code written in a high level language into machine code before execution.",
"Multithreading enables a program to execute multiple threads concurrently to improve performance.",
"Pointers store the memory address of another variable and are heavily used in languages like C.",
"An algorithm is a step by step procedure designed to perform a specific task or solve a problem.",
"Lambda functions are small anonymous functions defined without a name using a concise syntax.",
"String concatenation joins two or more text values together into a single string variable.",
"A conditional statement such as if else lets a program choose between different paths of execution.",
"Type casting converts a value from one data type to another such as integer to float.",
"An interpreter executes source code line by line without producing a separate machine code file.",
"Static typing requires variable types to be declared at compile time rather than inferred at runtime.",
"A class constructor initializes the properties of a newly created object when it is instantiated.",
"Operator overloading allows a programming language to redefine how operators behave for custom types.",
"A syntax error occurs when code violates the grammatical rules of a programming language.",
"Garbage collection automatically frees memory that is no longer referenced by any part of a program.",
"Command line arguments allow a program to accept input values when it is launched from a terminal.",
"A module or package groups related functions and classes together for reuse across projects.",
"Unit tests written in code verify that individual functions behave correctly for given inputs.",
"An event driven program responds to user actions such as clicks or key presses as they occur.",
"Immutable data types cannot be changed after they are created, such as tuples in Python.",
"A closure is a function that retains access to variables from its enclosing scope even after that scope has finished executing."
],
"Database": [
"A relational database organizes data into tables consisting of rows and columns linked by keys.",
"SQL stands for structured query language and is used to query and manipulate relational databases.",
"Normalization is the process of organizing data to reduce redundancy and improve data integrity.",
"A primary key uniquely identifies each record in a database table and cannot contain null values.",
"Indexes improve the speed of data retrieval operations on a database table at the cost of additional storage.",
"A foreign key is a column that creates a link between data in two different tables.",
"Transactions ensure that a series of database operations either all succeed or all fail together.",
"NoSQL databases like MongoDB store data in flexible document formats instead of rigid tables.",
"Joins combine rows from two or more tables based on a related column between them.",
"ACID properties atomicity consistency isolation and durability guarantee reliable database transactions.",
"A database schema defines the structure including tables fields and relationships within a database.",
"Views are virtual tables based on the result of a stored SQL query in a database.",
"Database sharding splits a large database into smaller faster more easily managed pieces called shards.",
"Stored procedures are precompiled SQL code that can be executed repeatedly on the database server.",
"Backup and recovery strategies protect databases against data loss due to hardware or software failure.",
"A composite key uses two or more columns together to uniquely identify a row in a table.",
"Query optimization improves the execution plan chosen by a database engine to run a SQL statement faster.",
"A trigger automatically executes a set of actions when a specified event occurs on a database table.",
"Denormalization intentionally introduces redundancy into a database to improve read performance.",
"An entity relationship diagram visually represents tables and their relationships during database design.",
"A database administrator manages user access, backups, and performance tuning for a database system.",
"Replication copies data from one database server to another to improve availability and fault tolerance.",
"The group by clause in SQL aggregates rows that share a common value into summary rows.",
"A data warehouse stores large volumes of historical data collected from multiple sources for analysis.",
"Referential integrity ensures that foreign key values always correspond to an existing primary key.",
"An object relational mapper lets developers interact with a database using programming language objects instead of raw SQL.",
"Concurrency control prevents conflicting simultaneous transactions from corrupting shared database data.",
"A cursor allows a program to iterate through the result set of a database query row by row.",
"Column oriented databases store data by column rather than by row to speed up analytical queries.",
"Database migration scripts apply incremental schema changes to keep a database structure up to date."
],
"Networking": [
"The internet protocol suite TCP IP defines how data is transmitted across interconnected networks.",
"A router forwards data packets between computer networks and determines the best path for data.",
"DNS translates human readable domain names into numerical IP addresses used by computers.",
"Bandwidth refers to the maximum rate of data transfer across a given network path.",
"The OSI model divides network communication into seven distinct layers from physical to application.",
"A subnet mask is used to divide an IP address into network and host portions.",
"Network topology describes the arrangement of nodes and connections in a computer network.",
"HTTP is an application layer protocol used for transmitting hypertext over the world wide web.",
"A switch connects devices within a local area network and forwards data based on MAC addresses.",
"Latency is the time delay between sending and receiving data across a network connection.",
"Wireless networking uses radio waves to connect devices without physical cables such as WiFi.",
"Network congestion occurs when too much data is sent through a network causing slowdowns.",
"A packet is a small unit of data transmitted over a network containing header and payload.",
"DHCP automatically assigns IP addresses to devices when they join a network.",
"A local area network connects computers within a limited geographic area such as an office building.",
"A wide area network spans a large geographic area connecting multiple local area networks together.",
"Network address translation allows multiple devices on a private network to share a single public IP address.",
"Full duplex communication allows data to be transmitted and received simultaneously over a network link.",
"A modem converts digital signals from a computer into analog signals suitable for transmission over telephone lines.",
"Ethernet is a wired networking technology commonly used to connect devices within a local area network.",
"Bluetooth is a short range wireless technology used for connecting peripheral devices to a computer or phone.",
"A proxy server acts as an intermediary that forwards client requests to other servers on a network.",
"Network throughput measures the actual amount of data successfully delivered over a network connection per second.",
"A domain name server resolves website names into the IP addresses computers use to locate each other.",
"Load balancing distributes incoming network traffic across multiple servers to prevent any single server from being overwhelmed.",
"A network interface card allows a computer to physically connect to a wired or wireless network.",
"Routing tables store information used by a router to determine the best path for forwarding packets.",
"A ping utility measures the round trip time it takes for a small packet to reach a remote host and return.",
"Fiber optic cables transmit data as pulses of light and support very high network bandwidth over long distances.",
"Quality of service settings prioritize certain types of network traffic such as video calls over other data."
],
"Operating System": [
"An operating system manages computer hardware and software resources and provides services to programs.",
"Process scheduling determines the order in which processes access the CPU for execution.",
"Virtual memory allows a computer to use disk space to extend available physical memory.",
"A deadlock occurs when two or more processes are unable to proceed because each waits for the other.",
"The kernel is the core component of an operating system that manages hardware resources directly.",
"Multiprogramming allows multiple programs to reside in memory and share the CPU efficiently.",
"File systems organize and store files on storage devices using directories and metadata.",
"Context switching is the process of storing and restoring the state of a process during scheduling.",
"Semaphores are synchronization tools used to control access to shared resources by multiple processes.",
"Paging is a memory management scheme that eliminates the need for contiguous allocation of memory.",
"An operating system provides an interface between the user and the computer hardware.",
"Thread management allows an operating system to execute multiple threads within a single process.",
"Interrupt handling allows the operating system to respond immediately to hardware or software events.",
"Device drivers allow the operating system to communicate with hardware peripherals effectively.",
"Thrashing occurs when a system spends more time swapping pages than executing actual processes.",
"A process control block stores information the operating system needs to manage a specific running process.",
"Round robin scheduling gives each process a fixed time slice before switching to the next process in the queue.",
"A shell provides a command line interface that lets users interact directly with the operating system.",
"Segmentation divides a program into logically distinct sections such as code, data, and stack for memory management.",
"System calls allow user programs to request services directly from the operating system kernel.",
"A boot loader loads the operating system kernel into memory when a computer is first powered on.",
"Demand paging loads pages of a process into memory only when they are actually needed during execution.",
"A zombie process has finished executing but still has an entry in the process table awaiting cleanup.",
"Priority scheduling assigns each process a priority level that determines its position in the execution queue.",
"Mutual exclusion ensures that only one process can access a critical section of shared resources at a time.",
"The dining philosophers problem is a classic example used to illustrate deadlock and synchronization issues.",
"Spooling allows multiple print jobs to be queued and processed by a printer without conflicting with each other.",
"A real time operating system guarantees that certain tasks are completed within strict time constraints.",
"Fragmentation occurs when free memory is broken into small non contiguous blocks over time.",
"An operating system scheduler decides which ready process in memory gets access to the CPU next."
],
"Artificial Intelligence": [
"Artificial intelligence is the simulation of human intelligence processes by computer systems.",
"Machine learning enables systems to learn patterns from data without being explicitly programmed.",
"Neural networks are computing systems inspired by the biological neural networks of the human brain.",
"Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
"Unsupervised learning finds hidden patterns or groupings in data without labeled outcomes.",
"Reinforcement learning trains an agent to make decisions by rewarding desired behaviors.",
"Natural language processing allows computers to understand interpret and generate human language.",
"Computer vision enables machines to interpret and understand visual information from images or video.",
"A decision tree is a flowchart like structure used for classification and regression tasks.",
"Deep learning uses multilayer neural networks to automatically extract features from raw data.",
"Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models.",
"An expert system uses a knowledge base and inference rules to solve complex problems like a human expert.",
"Overfitting occurs when a machine learning model learns noise in training data instead of general patterns.",
"Feature engineering involves selecting and transforming variables to improve model performance.",
"Genetic algorithms are search heuristics inspired by the process of natural selection and evolution.",
"A convolutional neural network is particularly effective at processing grid like data such as images.",
"A recurrent neural network processes sequential data by maintaining a hidden state across time steps.",
"Transfer learning reuses a model trained on one task as the starting point for a related task.",
"Clustering groups similar data points together without relying on predefined category labels.",
"An activation function introduces non linearity into a neural network allowing it to learn complex patterns.",
"A confusion matrix summarizes the correct and incorrect predictions made by a classification model.",
"Hyperparameter tuning searches for the best configuration settings to optimize a machine learning model.",
"A chatbot uses natural language processing techniques to simulate human like conversation with users.",
"Speech recognition converts spoken audio input into written text using trained acoustic models.",
"An autonomous vehicle uses artificial intelligence to perceive its environment and make driving decisions.",
"A recommendation system predicts items a user is likely to prefer based on past behavior and preferences.",
"Ensemble learning combines predictions from multiple models to produce a more accurate overall result.",
"Backpropagation calculates gradients used to update the weights of a neural network during training.",
"A generative adversarial network uses two competing neural networks to generate realistic synthetic data.",
"Sentiment analysis uses natural language processing to determine the emotional tone expressed in text."
],
"Compiler Design": [
"A compiler translates source code from a high level language into an equivalent target language.",
"Lexical analysis is the first phase of compilation that converts source code into a stream of tokens.",
"Syntax analysis or parsing checks whether the sequence of tokens conforms to the grammar of the language.",
"Semantic analysis checks the source code for meaning and type consistency after parsing.",
"An abstract syntax tree represents the hierarchical syntactic structure of source code.",
"Code optimization improves the intermediate representation of a program to make it more efficient.",
"Code generation is the final phase of compilation that produces target machine code.",
"A symbol table stores information about identifiers such as variables functions and their scope.",
"Context free grammars are used to define the syntax rules of a programming language.",
"A parser generator automatically creates a parser from a formal grammar specification.",
"Intermediate code representation acts as a bridge between source code and target machine code.",
"Lexical analyzers use finite automata to recognize patterns and generate tokens from source code.",
"Top down parsing builds the parse tree from the root down to the leaves of the syntax tree.",
"Bottom up parsing builds the parse tree starting from the leaves and working up to the root.",
"A compiler performs error detection and reporting during each phase of the compilation process.",
"A regular expression describes a pattern used by a lexical analyzer to identify valid tokens.",
"A finite state automaton models the states and transitions used during lexical scanning of source code.",
"Type checking during semantic analysis verifies that operations are applied to compatible data types.",
"Peephole optimization examines a small window of instructions to eliminate redundant operations.",
"A shift reduce parser uses a stack to decide whether to shift a token or reduce a grammar rule.",
"An LL parser reads input from left to right and constructs a leftmost derivation of the grammar.",
"An LR parser reads input from left to right and constructs a rightmost derivation in reverse.",
"Dead code elimination removes instructions from a program that do not affect the final output.",
"A three address code representation expresses each instruction using at most three operands.",
"Register allocation assigns a limited number of physical registers to a larger set of program variables.",
"A recursive descent parser implements each grammar rule as a separate mutually recursive function.",
"Scope resolution during compilation determines which declaration a given identifier reference refers to.",
"A macro preprocessor expands shorthand code definitions before the main compilation phases begin.",
"Loop unrolling is an optimization technique that reduces loop overhead by duplicating the loop body.",
"An interpreter differs from a compiler because it executes source code directly without producing machine code."
],
"Software Engineering": [
"Software engineering applies engineering principles to the design development and maintenance of software.",
"The software development life cycle defines the phases involved in building software applications.",
"Agile methodology emphasizes iterative development collaboration and flexibility in software projects.",
"Requirements gathering involves collecting and documenting what stakeholders expect from a software system.",
"Software testing verifies that a system behaves as expected and identifies defects before release.",
"Version control and configuration management track changes to software artifacts over time.",
"Design patterns are reusable solutions to commonly occurring problems in software design.",
"Software architecture defines the high level structure and organization of a software system.",
"Unit testing verifies the correctness of individual components or functions in isolation.",
"The waterfall model is a sequential software development process divided into distinct phases.",
"Code review is the practice of systematically examining source code to find bugs and improve quality.",
"Software maintenance involves modifying a software product after delivery to fix issues or add features.",
"Scrum is an agile framework that organizes work into fixed length iterations called sprints.",
"Technical debt refers to the implied cost of additional rework caused by choosing quick solutions.",
"Continuous integration automatically builds and tests code changes as they are committed to a repository.",
"A use case diagram documents the interactions between users and a software system during requirements analysis.",
"Software project management involves planning, scheduling, and tracking resources to deliver a project on time.",
"Refactoring restructures existing code to improve its readability and maintainability without changing its behavior.",
"A stakeholder is any person or group with an interest in the outcome of a software project.",
"Integration testing verifies that different modules of a software system work correctly together.",
"A user story describes a software feature from the perspective of an end user in simple language.",
"Software quality assurance establishes processes to ensure a product meets defined standards before release.",
"A sprint retrospective is a meeting held at the end of an agile iteration to reflect on what went well.",
"Requirements elicitation gathers information from stakeholders to define what a software system must do.",
"A feasibility study evaluates whether a proposed software project is technically and financially viable.",
"Pair programming involves two developers working together at one workstation, one writing code and one reviewing it.",
"A software requirements specification document formally describes the expected behavior of a system to be built.",
"Regression testing re-executes existing tests to ensure new code changes have not broken previous functionality.",
"The spiral model combines iterative development with systematic risk analysis at each phase of a project.",
"A burndown chart visually tracks the amount of work remaining in an agile sprint over time."
],
"Cyber Security": [
"Cyber security involves protecting computer systems networks and data from digital attacks.",
"Encryption converts readable data into an encoded format that can only be read with a decryption key.",
"A firewall acts as a barrier between a trusted internal network and untrusted external networks.",
"Phishing is a social engineering attack that tricks users into revealing sensitive information such as passwords.",
"Malware is malicious software designed to damage disrupt or gain unauthorized access to computer systems.",
"Two factor authentication adds an extra layer of security by requiring a second verification step beyond a password.",
"A vulnerability is a weakness in a system that can be exploited by an attacker to cause harm.",
"Penetration testing simulates cyber attacks to identify security weaknesses before real attackers exploit them.",
"A denial of service attack overwhelms a system with malicious traffic making it unavailable to legitimate users.",
"Ransomware encrypts a victim's files and demands payment in exchange for the decryption key.",
"Access control restricts who can view or use sensitive resources based on defined security policies.",
"Intrusion detection systems monitor for suspicious activity and known attack signatures on a network.",
"A security audit systematically evaluates an organization's information systems to find compliance gaps and weaknesses.",
"Zero day exploits target software vulnerabilities that are unknown to the vendor at the time of the attack.",
"Social engineering manipulates people psychologically into divulging confidential information or performing risky actions.",
"A digital signature verifies the authenticity and integrity of a message using public key cryptography.",
"SQL injection is an attack technique that inserts malicious database queries through unvalidated user input fields.",
"A brute force attack systematically tries every possible password combination until the correct one is found.",
"Antivirus software scans files and programs to detect and remove known malicious code signatures.",
"A security patch is a software update released specifically to fix a discovered vulnerability.",
"Cross site scripting injects malicious scripts into trusted websites to steal data from unsuspecting visitors.",
"A honeypot is a decoy system deliberately exposed to attract and study attacker behavior.",
"Data breach incidents occur when confidential information is accessed or stolen without authorization.",
"A keylogger secretly records every keystroke a victim types in order to steal credentials.",
"Public key infrastructure manages digital certificates that establish trust between parties over insecure networks.",
"A spoofing attack disguises malicious communication so it appears to come from a trusted source.",
"Security incident response defines the procedures an organization follows immediately after a cyber attack is detected.",
"Multi layered defense combines several independent security controls so that a single failure does not compromise the whole system.",
"A trojan horse disguises malicious code as legitimate software to trick users into installing it.",
"Endpoint protection software secures individual devices such as laptops and phones against malware and unauthorized access."
],
"Data Structure": [
"A data structure is a specialized format for organizing processing retrieving and storing data.",
"An array stores a fixed size sequential collection of elements of the same data type.",
"A linked list consists of nodes where each node contains data and a reference to the next node.",
"A stack is a linear data structure that follows the last in first out principle.",
"A queue is a linear data structure that follows the first in first out principle.",
"A binary tree is a hierarchical structure where each node has at most two children.",
"A hash table stores key value pairs and uses a hash function to compute an index into an array.",
"A graph consists of a set of vertices connected by edges representing relationships between entities.",
"Binary search trees maintain elements in sorted order to allow efficient search insertion and deletion.",
"Sorting algorithms like quicksort and mergesort arrange elements in a particular order efficiently.",
"A heap is a specialized tree based structure that satisfies the heap property for priority queues.",
"Time complexity measures how the runtime of an algorithm grows with the size of its input.",
"A doubly linked list allows traversal in both forward and backward directions using two pointers.",
"Depth first search explores as far as possible along a branch before backtracking in a graph.",
"Breadth first search explores all neighbor nodes at the current depth before moving to the next level.",
"A circular queue connects the last position back to the first to efficiently reuse empty slots.",
"A trie is a tree like data structure used to efficiently store and search strings sharing common prefixes.",
"Space complexity measures the amount of memory an algorithm requires relative to its input size.",
"A balanced tree such as an AVL tree automatically adjusts its structure to keep operations efficient.",
"Dynamic programming solves complex problems by breaking them into overlapping subproblems and storing results.",
"A priority queue serves elements based on their priority value rather than their insertion order.",
"An adjacency matrix represents graph connections using a two dimensional array of zeros and ones.",
"A binary search algorithm repeatedly divides a sorted array in half to quickly locate a target value.",
"A collision in a hash table occurs when two different keys map to the same index location.",
"Amortized analysis evaluates the average time complexity of an operation performed repeatedly over a sequence of operations.",
"A red black tree is a self balancing binary search tree that maintains logarithmic height through coloring rules.",
"An array based implementation of a stack uses a fixed size array with a pointer tracking the top element.",
"A linked list based queue allows dynamic growth in size without needing to predefine a maximum capacity.",
"Tree traversal methods such as inorder preorder and postorder visit nodes of a tree in different sequences.",
"A skip list uses multiple layers of linked lists to achieve fast search performance similar to balanced trees."
],
"Computer Architecture": [
"Computer architecture defines the functional behavior and organization of a computer system.",
"The central processing unit executes instructions by performing arithmetic logical and control operations.",
"Cache memory is a small fast memory located close to the CPU to speed up data access.",
"The instruction set architecture defines the set of instructions a processor can execute.",
"Pipelining allows multiple instruction stages to overlap improving overall processor throughput.",
"RAM or random access memory provides temporary fast storage for data actively used by programs.",
"The control unit directs the operation of the processor by interpreting and executing instructions.",
"A bus is a communication system that transfers data between components inside a computer.",
"Registers are small high speed storage locations directly inside the processor used for computation.",
"The arithmetic logic unit performs mathematical and logical operations within the processor.",
"Von Neumann architecture uses a single memory space to store both instructions and data.",
"Parallel processing uses multiple processors simultaneously to perform computations faster.",
"Clock speed measures how many cycles a processor can execute per second typically in gigahertz.",
"Memory hierarchy organizes storage into levels balancing speed cost and capacity from registers to disk.",
"A microprocessor is an integrated circuit that contains the functions of a central processing unit.",
"Harvard architecture uses separate memory spaces for instructions and data to allow simultaneous access.",
"A hardwired control unit implements control signals using fixed digital logic circuits rather than software.",
"Instruction level parallelism allows a processor to execute multiple instructions during a single clock cycle.",
"A multicore processor contains multiple independent processing units on a single physical chip.",
"Superscalar architecture allows a processor to issue and execute more than one instruction per clock cycle.",
"Branch prediction guesses the outcome of a conditional instruction to keep the processor pipeline full.",
"A cache miss occurs when requested data is not found in cache memory and must be fetched from main memory.",
"Reduced instruction set computing uses a small set of simple instructions to simplify processor design.",
"Complex instruction set computing allows a single instruction to perform multiple low level operations.",
"An addressing mode specifies how the operand of an instruction is located within computer memory.",
"A microcontroller integrates a processor, memory, and input output peripherals onto a single chip.",
"Direct memory access allows peripheral devices to transfer data to memory without involving the CPU directly.",
"Overclocking increases a processor's clock speed beyond its rated specification to boost performance.",
"A bus width determines how many bits of data can be transferred simultaneously between components.",
"Firmware is low level software permanently programmed into hardware to control basic device operations."
]
}

# ============================================================
# Category-specific technical term banks (for template augmentation)
# ============================================================
terms = {
"Programming": ["recursion","polymorphism","inheritance","exception handling","multithreading","pointers",
    "garbage collection","closures","lambda functions","type casting","operator overloading","syntax errors",
    "debugging","version control","compilers","interpreters","static typing","dynamic typing","class constructors",
    "command line arguments","modules","unit testing","event driven programming","immutable data types",
    "string concatenation","conditional statements","loops","arrays","algorithms","variables"],
"Database": ["normalization","primary keys","foreign keys","indexes","transactions","ACID properties","joins",
    "views","stored procedures","triggers","sharding","replication","denormalization","entity relationship diagrams",
    "query optimization","referential integrity","object relational mapping","concurrency control","cursors",
    "data warehouses","composite keys","database schemas","NoSQL databases","column oriented databases",
    "database migrations","backup and recovery","database administrators","SQL queries","relational tables","database views"],
"Networking": ["routers","DNS","bandwidth","subnet masks","network topology","the HTTP protocol","switches","latency",
    "wireless networking","packets","DHCP","local area networks","wide area networks","network address translation",
    "full duplex communication","modems","ethernet","bluetooth","proxy servers","network throughput","domain name servers",
    "load balancing","network interface cards","routing tables","ping utilities","fiber optic cables",
    "quality of service","the TCP IP protocol","the OSI model","network congestion"],
"Operating System": ["process scheduling","virtual memory","deadlocks","kernels","multiprogramming","file systems",
    "context switching","semaphores","paging","thread management","interrupt handling","device drivers","thrashing",
    "process control blocks","round robin scheduling","shells","segmentation","system calls","boot loaders",
    "demand paging","zombie processes","priority scheduling","mutual exclusion","spooling",
    "real time operating systems","fragmentation","schedulers","memory management","multitasking","process synchronization"],
"Artificial Intelligence": ["machine learning","neural networks","supervised learning","unsupervised learning",
    "reinforcement learning","natural language processing","computer vision","decision trees","deep learning",
    "gradient descent","expert systems","overfitting","feature engineering","genetic algorithms",
    "convolutional neural networks","recurrent neural networks","transfer learning","clustering","activation functions",
    "confusion matrices","hyperparameter tuning","chatbots","speech recognition","autonomous vehicles",
    "recommendation systems","ensemble learning","backpropagation","generative adversarial networks",
    "sentiment analysis","artificial neural networks"],
"Compiler Design": ["lexical analysis","syntax analysis","semantic analysis","abstract syntax trees","code optimization",
    "code generation","symbol tables","context free grammars","parser generators","intermediate code",
    "finite automata","top down parsing","bottom up parsing","regular expressions","type checking",
    "peephole optimization","shift reduce parsers","LL parsers","LR parsers","dead code elimination",
    "three address code","register allocation","recursive descent parsers","scope resolution","macro preprocessors",
    "loop unrolling","interpreters","compilers","tokenization","grammar rules"],
"Software Engineering": ["the software development life cycle","agile methodology","requirements gathering",
    "software testing","version control","design patterns","software architecture","unit testing","the waterfall model",
    "code review","software maintenance","scrum","technical debt","continuous integration","use case diagrams",
    "project management","refactoring","stakeholders","integration testing","user stories","quality assurance",
    "sprint retrospectives","requirements elicitation","feasibility studies","pair programming",
    "requirements specifications","regression testing","the spiral model","burndown charts","software prototyping"],
"Cyber Security": ["encryption","firewalls","phishing","malware","two factor authentication","vulnerabilities",
    "penetration testing","denial of service attacks","ransomware","access control","intrusion detection",
    "security audits","zero day exploits","social engineering","digital signatures","SQL injection",
    "brute force attacks","antivirus software","security patches","cross site scripting","honeypots",
    "data breaches","keyloggers","public key infrastructure","spoofing attacks","incident response",
    "multi layered defense","trojan horses","endpoint protection","cyber attacks"],
"Data Structure": ["arrays","linked lists","stacks","queues","binary trees","hash tables","graphs",
    "binary search trees","sorting algorithms","heaps","time complexity","doubly linked lists","depth first search",
    "breadth first search","circular queues","tries","space complexity","balanced trees","dynamic programming",
    "priority queues","adjacency matrices","binary search","hash collisions","amortized analysis","red black trees",
    "tree traversal","skip lists","linked list implementations","array implementations","graph algorithms"],
"Computer Architecture": ["central processing units","cache memory","instruction set architecture","pipelining",
    "random access memory","control units","buses","registers","arithmetic logic units","von neumann architecture",
    "parallel processing","clock speed","memory hierarchy","microprocessors","harvard architecture",
    "hardwired control units","instruction level parallelism","multicore processors","superscalar architecture",
    "branch prediction","cache misses","reduced instruction set computing","complex instruction set computing",
    "addressing modes","microcontrollers","direct memory access","overclocking","bus width","firmware",
    "digital logic circuits"]
}

# ============================================================
# Generic templates (NO category name mentioned -> avoids label leakage)
# Each combines TWO technical terms for richer signal per sentence.
# ============================================================
templates = [
    "{t1} and {t2} are both topics that appear frequently in advanced coursework.",
    "A strong grasp of {t1} often makes it easier to later understand {t2}.",
    "Textbooks commonly cover {t1} in the same chapter that introduces {t2}.",
    "Engineers who work with {t1} also frequently need to understand {t2}.",
    "Exam questions often test whether students can relate {t1} to {t2}.",
    "{t1} is sometimes compared with {t2} to highlight their key differences.",
    "Research papers have explored how {t1} interacts with {t2} in practice.",
    "A project involving {t1} may also require careful handling of {t2}.",
    "Lecture notes often introduce {t1} before moving on to {t2}.",
    "Professionals frequently combine {t1} with {t2} to solve real problems.",
    "{t1} builds a useful foundation for later learning about {t2}.",
    "Some interview questions ask candidates to explain both {t1} and {t2}.",
]

def make_sentence(t1, t2):
    t1c = t1[0].upper() + t1[1:] if not t1[0].isupper() else t1
    tmpl = random.choice(templates)
    sentence = tmpl.format(t1=t1c, t2=t2)
    return sentence

rows = []
for category, sentences in base_data.items():
    # 30 hand-written originals
    for s in sentences:
        rows.append((s, category))

    # 220 template-augmented sentences from term pairs (250 total per category)
    term_list = terms[category]
    pairs_needed = 220
    generated = set()
    attempts = 0
    while len(generated) < pairs_needed and attempts < 8000:
        attempts += 1
        t1, t2 = random.sample(term_list, 2)
        sent = make_sentence(t1, t2)
        if sent not in generated:
            generated.add(sent)
    for s in generated:
        rows.append((s, category))

random.shuffle(rows)

with open("dataset/dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "category"])
    for text, cat in rows:
        writer.writerow([text, cat])

print("Dataset generated:", len(rows), "rows")
import collections
counts = collections.Counter(cat for _, cat in rows)
for cat, n in sorted(counts.items()):
    print(f"  {cat}: {n}")
