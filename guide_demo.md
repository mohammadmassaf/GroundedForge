# Study Guide — TCP reliability: sequence numbers, acknowledgements and retransmission

## Introduction to TCP Reliability

- TCP uses sequence numbers and acknowledgments to make transmission reliable

> 📖 `rfc793_p16_c0` — rfc793.txt, p.16: “September 1981 Transmission Control Protocol Philosophy      Transmission is made reliable…”
- Each octet of data is assigned a sequence number

> 📖 `rfc793_p16_c0` — rfc793.txt, p.16: “September 1981 Transmission Control Protocol Philosophy      Transmission is made reliable…”
> 📖 `rfc793_p30_c0` — rfc793.txt, p.30: “September 1981 Transmission Control Protocol Functional Specification    3.3.  Sequence Nu…”
## Sequence Numbers

- The sequence number of the first octet of data in a segment is transmitted with that segment

> 📖 `rfc793_p16_c0` — rfc793.txt, p.16: “September 1981 Transmission Control Protocol Philosophy      Transmission is made reliable…”
- The acknowledgment mechanism is cumulative, indicating that all octets up to but not including the acknowledged sequence number have been received

> 📖 `rfc793_p30_c0` — rfc793.txt, p.30: “September 1981 Transmission Control Protocol Functional Specification    3.3.  Sequence Nu…”
- Sequence numbers are used to correctly order segments that may be received out of order and to eliminate duplicates

> 📖 `rfc793_p10_c2` — rfc793.txt, p.10: “to each octet     transmitted, and requiring a positive acknowledgment (ACK) from the     receiving TCP.  If the ACK is not received within a timeout…”
## Acknowledgments

- Segments carry an acknowledgment number which is the sequence number of the next expected data octet of transmissions in the reverse direction

> 📖 `rfc793_p16_c0` — rfc793.txt, p.16: “September 1981 Transmission Control Protocol Philosophy      Transmission is made reliable…”
- The receiving TCP sends an acknowledgment showing its next expected sequence number even when the window is zero

> 📖 `rfc793_p48_c3` — rfc793.txt, p.48: “selves, but will be prepared for such behavior   on the part of other TCPs.    The sending TCP must be prepared to accept from the user and send at…”
## Retransmission

- TCP uses retransmission to ensure delivery of every segment

> 📖 `rfc793_p46_c2` — rfc793.txt, p.46: “n a non-secure environment   (the values would indicate unclassified data), thus hosts in   non-secure environments must be prepared to receive the se…”
> 📖 `rfc793_p10_c2` — rfc793.txt, p.10: “to each octet     transmitted, and requiring a positive acknowledgment (ACK) from the     receiving TCP.  If the ACK is not received within a timeout…”
- The sending TCP must regularly retransmit to the receiving TCP even when the window is zero

> 📖 `rfc793_p48_c3` — rfc793.txt, p.48: “selves, but will be prepared for such behavior   on the part of other TCPs.    The sending TCP must be prepared to accept from the user and send at…”
- Retransmissions of sequence numbers between SND.UNA and SND.NXT are expected

> 📖 `rfc793_p89_c2` — rfc793.txt, p.89: “window field specified in segments from the remote (data           receiving) TCP.  The range of new sequence numbers which may           be emitted…”