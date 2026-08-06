# Demo corpus

Three IETF RFCs, included so this repo is runnable from a clean clone without
supplying any material of your own:

| file | RFC | topic |
|---|---|---|
| `rfc768.txt` | [RFC 768](https://www.rfc-editor.org/rfc/rfc768.txt) | User Datagram Protocol (UDP) |
| `rfc791.txt` | [RFC 791](https://www.rfc-editor.org/rfc/rfc791.txt) | Internet Protocol (IP) |
| `rfc793.txt` | [RFC 793](https://www.rfc-editor.org/rfc/rfc793.txt) | Transmission Control Protocol (TCP) |

Retrieved verbatim from `www.rfc-editor.org`. RFCs are published by the IETF for
free distribution and reproduction; each file keeps its original text and
notices unaltered.

## Why these

They are plain text (no OCR, matching the digital-inputs-only constraint), they
are paginated with form-feed characters so citations land on a real page number,
and they cover the same ground as an introductory networking course — headers,
delays, connection setup, reliability — so the generated artifacts look like the
real thing rather than a toy.

## Use

```bash
python main.py ingest --corpus demo
python main.py build-index --corpus demo
python main.py make-quiz "TCP connection establishment" --corpus demo -n 5
```

Sample outputs generated from exactly this corpus: [quiz_demo.md](../quiz_demo.md),
[guide_demo.md](../guide_demo.md).

The eval numbers reported in the top-level README are measured on a private
course corpus, not on this one.
