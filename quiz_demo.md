# Quiz — TCP connection establishment and the three-way handshake

### Q1. What is the procedure used to establish a connection in TCP?

**Answer:** The three-way handshake

> 📖 `rfc793_p36_c3` — rfc793.txt, p.36: “on    The "three-way handshake" is the procedure used to establish a   connection.  This procedure normally is initiated by one TCP and   responded to…”
> 📖 `rfc793_p36_c2` — rfc793.txt, p.36: “To summarize: every segment emitted occupies one or more sequence     numbers in the sequence space, the numbers occupied by a segment are     "…”
> 📖 `rfc793_p18_c0` — rfc793.txt, p.18: “September 1981 Transmission Control Protocol Philosophy      If there are several pending…”

### Q2. What is the purpose of the three-way handshake in TCP?

**Answer:** To prevent old duplicate connection initiations from causing confusion

> 📖 `rfc793_p38_c1` — rfc793.txt, p.38: “--> SYN-RECEIVED    5.  SYN-RECEIVED --> <SEQ=100><ACK=301><CTL=SYN,ACK> ...    6.  ESTABLISHED  <-- <SEQ=300><ACK=101><CTL=SYN,ACK> <-- SYN-REC…”

### Q3. How many messages are exchanged during the three-way handshake?

**Answer:** Three

> 📖 `rfc793_p18_c0` — rfc793.txt, p.18: “September 1981 Transmission Control Protocol Philosophy      If there are several pending…”
> 📖 `rfc793_p37_c0` — rfc793.txt, p.37: “September 1981                                                                                                      Transmission Control Protocol…”

### Q4. What control flag is used to initiate a connection in TCP?

**Answer:** SYN

> 📖 `rfc793_p18_c0` — rfc793.txt, p.18: “September 1981 Transmission Control Protocol Philosophy      If there are several pending…”
> 📖 `rfc793_p37_c2` — rfc793.txt, p.37: “TCP B    1.  CLOSED                                               LISTEN    2.  SYN-SENT    --> <SEQ=100>…”

### Q5. What happens when a TCP is in a non-synchronized state and receives an acceptable reset?

**Answer:** It returns to LISTEN

> 📖 `rfc793_p38_c1` — rfc793.txt, p.38: “--> SYN-RECEIVED    5.  SYN-RECEIVED --> <SEQ=100><ACK=301><CTL=SYN,ACK> ...    6.  ESTABLISHED  <-- <SEQ=300><ACK=101><CTL=SYN,ACK> <-- SYN-REC…”
