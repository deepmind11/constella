"""Eval harness for Constella.

Inspired by Hippocratic's RWE-LLM 4-stage safety validation framework
(medRxiv 10.1101/2025.03.17.25324157). Five rubric dimensions per turn,
0-3 each except latency_pass which is bool.
"""
