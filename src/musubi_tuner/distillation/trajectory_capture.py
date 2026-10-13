def get_capture_indices(total_steps: int, num_captures: int) -> list[int]:
    """Return indices of steps to capture, uniformly spaced.
    
    Example: total_steps=15, num_captures=5 -> [0, 3, 7, 11, 14]
    """
    if num_captures >= total_steps:
        return list(range(total_steps))
    if num_captures <= 1:
        return [total_steps - 1] if total_steps > 0 else []
        
    return [round(i * (total_steps - 1) / (num_captures - 1)) for i in range(num_captures)]
