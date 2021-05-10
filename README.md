# Auto-Encoder for Anomaly Detection

A system that consists of one legitimate user and one eavesdropper is considered.

Conventionally, the legitimate user (Bob) sends a signal to a base station (BS) to request access to the network. In the case that Eve keeps silent and the BS authenticates Bob's signal correctly, then the signal will be labelled as NORMAL.

However, if Eve impersonates Bob, then the BS must be capable of detecting if its received signal is associated with Eve or not. In the case the detection is correct, then the signal received at the BS will be labelled as ANOMALOUS. By contrast, the signal will be labelled NORMAL, in this case, we have false detection because the received signal (constituted by Bob's transmitted signal and Eve's transmitted signal) is misunderstood as a normal signal.

---
The goal is to detect if the signal received at the BS is associated with a spoofing attack or not.

---
Styles:
- Object-oriented programming
- PEP 8
