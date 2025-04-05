class TrieNode:
    def __init__(self, token=None):
        self.token = token
        self.children = {} # All child nodes
        self.logprob_to_generate_any_forbidden_sample_given_current_prefix = None # This is log P(forbidden completion | current prefix)

def insert_forbidden_sequence(root: TrieNode, tokens, logprobs):
    """
    Insert a forbidden sequence (tokens + logprobs) into the trie.

    :param root:     The root TrieNode.
    :param tokens:   List of tokens, e.g. ['I', 'love', 'NLP'].
    :param logprobs: List of float log-probabilities for each token.
    """

    # --- Step 1: Compute cumulative sum of logprobs from right ---
    # cumsum[i] = logprobs[i+1] + logprobs[i+2] + ... + logprobs[n-1] + 0.0
    logprobs = np.array(logprobs, dtype=np.float64)  # Ensure consistent precision
    logprobs = np.append(logprobs, 0.0) 
    cumsum = np.cumsum(logprobs[::-1])[::-1]
    cumsum = cumsum[1:]

    # --- Step 2: Traverse the trie from the root, token by token ---
    node = root
    for i, token in enumerate(tokens):
        # If this token doesn't exist yet, create a child node
        if token not in node.children:
            node.children[token] = TrieNode(token=token)

        # Move down to the child
        node = node.children[token]

        # --- Step 3: Update the node's logprob value ---
        # Use numpy's logsumexp for numerical stability
        current_logprob = node.logprob_to_generate_any_forbidden_sample_given_current_prefix
        try:
            node.logprob_to_generate_any_forbidden_sample_given_current_prefix = (
                np.logaddexp(current_logprob, cumsum[i])
                if current_logprob is not None else cumsum[i]
            )
        except:
            import pdb; pdb.set_trace()

def adjust_next_token_log_probs(node, next_token_log_probs):
    """
    Adjust the LM's next_token_log_probs in-place, subtracting out
    any mass that leads to forbidden completions from this trie node.

    :param node: TrieNode for the current prefix. Must have `children` and 
                 `logprob_to_generate_any_forbidden_sample_given_current_prefix` attributes.
    :param next_token_log_probs: torch.Tensor
        A tensor containing log-probabilities of tokens. This is modified in-place.
    """
    if node is None or not hasattr(node, "children") or len(node.children) == 0:
        # No trie node or children to process
        return

    def logminusexp(a, b):
        """
        Numerically stable computation of log(exp(a) - exp(b)) assuming a > b.
        
        :param a: Logarithm of a positive number.
        :param b: Logarithm of a smaller positive number.
        :return: Logarithm of (exp(a) - exp(b)).
        """
        if b == float('-inf'):
            return a  # exp(b) is effectively zero
        if a <= b:
            # Handle cases where a <= b gracefully instead of raising an exception.
            return float('-inf')  # Log of a non-positive value is undefined
        return a + torch.log1p(-torch.exp(b - a))

    for token, child_node in node.children.items():
        lp_forbidden = getattr(child_node, 
                               "logprob_to_generate_any_forbidden_sample_given_current_prefix", 
                               None)
        if lp_forbidden is not None:
            # Ensure the forbidden log-prob is a valid number
            if not isinstance(lp_forbidden, (float, int)) or lp_forbidden == float('-inf'):
                continue  # Skip invalid or irrelevant values

            lp_token_given_prefix = next_token_log_probs[token]
            lp_to_remove = lp_token_given_prefix + lp_forbidden

            # Subtract log-probabilities using logminusexp
            next_token_log_probs[token] = logminusexp(lp_token_given_prefix, lp_to_remove)

    # Renormalize the log-probabilities
    log_total_mass = torch.logsumexp(next_token_log_probs, dim=0)
    if log_total_mass > float('-inf'):
        next_token_log_probs -= log_total_mass