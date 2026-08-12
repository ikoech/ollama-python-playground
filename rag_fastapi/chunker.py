# chunker.py
import re

def split_into_sentences(text):
    # naive sentence splitter that keeps punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, max_words=200, overlap_words=30):
    """
    Chunk text into pieces of roughly max_words words with overlap.
    This is simple, robust, and avoids external tokenizers.
    """
    sentences = split_into_sentences(text)
    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        words = sent.split()
        wcount = len(words)
        if current_words + wcount <= max_words:
            current.append(sent)
            current_words += wcount
        else:
            if current:
                chunks.append(" ".join(current))
            # start new chunk; allow very long sentence to be its own chunk
            current = []
            # if sentence itself longer than max_words, split by words
            if wcount > max_words:
                for i in range(0, wcount, max_words - overlap_words):
                    part = " ".join(words[i:i + (max_words - overlap_words)])
                    chunks.append(part)
                current = []
                current_words = 0
            else:
                current = [sent]
                current_words = wcount

    if current:
        chunks.append(" ".join(current))

    # add overlap between chunks
    if overlap_words > 0 and len(chunks) > 1:
        overlapped = []
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
                continue
            prev_words = overlapped[-1].split()
            cur_words = c.split()
            # take last overlap_words from prev and prepend to current
            overlap = prev_words[-overlap_words:] if len(prev_words) >= overlap_words else prev_words
            new_chunk = " ".join(overlap + cur_words)
            overlapped.append(new_chunk)
        return overlapped

    return chunks
