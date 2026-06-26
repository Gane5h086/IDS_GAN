"""Optional Scapy traffic generator for the IDS demo.

This script creates simple, controlled traffic patterns that can be run on the
attacker machine when you want something closer to real LAN traffic than a
pure UI event.

Examples:
    python scapy_sender.py --scenario dos --target 192.168.1.10
    python scapy_sender.py --scenario probe --target 192.168.1.10 --port-start 1 --port-end 50
    python scapy_sender.py --scenario brute_force --target 192.168.1.10 --count 200
"""

from __future__ import annotations

import argparse
import time

from scapy.all import ICMP, IP, Raw, TCP, send  # type: ignore


def send_dos(target: str, count: int, delay: float) -> None:
    packet = IP(dst=target) / ICMP() / Raw(load=b'DOS-DEMO')
    for _ in range(count):
        send(packet, verbose=False)
        time.sleep(delay)


def send_probe(target: str, port_start: int, port_end: int, delay: float) -> None:
    for port in range(port_start, port_end + 1):
        packet = IP(dst=target) / TCP(dport=port, flags='S')
        send(packet, verbose=False)
        time.sleep(delay)


def send_bruteforce(target: str, port: int, count: int, delay: float) -> None:
    for _ in range(count):
        packet = IP(dst=target) / TCP(dport=port, flags='S')
        send(packet, verbose=False)
        time.sleep(delay)


def send_repeated_http(target: str, port: int, count: int, delay: float) -> None:
    payload = b'GET / HTTP/1.1\r\nHost: demo\r\nConnection: close\r\n\r\n'
    for _ in range(count):
        packet = IP(dst=target) / TCP(dport=port, flags='PA') / Raw(load=payload)
        send(packet, verbose=False)
        time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate simple IDS demo traffic with Scapy.')
    parser.add_argument('--scenario', required=True, choices=['dos', 'probe', 'r2l', 'u2r', 'brute_force'])
    parser.add_argument('--target', required=True, help='Target IP address on the LAN')
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--delay', type=float, default=0.02)
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--port-start', type=int, default=1)
    parser.add_argument('--port-end', type=int, default=50)

    args = parser.parse_args()

    if args.scenario == 'dos':
        send_dos(args.target, args.count, args.delay)
    elif args.scenario == 'probe':
        send_probe(args.target, args.port_start, args.port_end, args.delay)
    elif args.scenario == 'brute_force':
        send_bruteforce(args.target, args.port, args.count, args.delay)
    elif args.scenario in {'r2l', 'u2r'}:
        send_repeated_http(args.target, args.port, args.count, args.delay)

    print(f'Completed {args.scenario} demo traffic against {args.target}')


if __name__ == '__main__':
    main()
