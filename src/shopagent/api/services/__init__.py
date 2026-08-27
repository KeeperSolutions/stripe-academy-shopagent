"""Commerce logic, with no HTTP in it (D6, D8).

The mirror of `api/routers/`: those modules speak HTTP and nothing else, these
speak carts and orders and nothing else. The split is the same one
`api/lifecycle.py` makes and for the same reason — D8's Stripe webhook and D9's
agent tools call these functions outside any request, where `HTTPException`
would have nobody to catch it. A module in here that imports FastAPI is a bug.
"""
