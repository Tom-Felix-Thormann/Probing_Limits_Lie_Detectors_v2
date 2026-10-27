# Let all 6 LLMs perform a forced choice task on the manual dataset in three conditions:
# LIE condition: LLMs are instructed to lie, options are true answer, false answer, misleading answer, none fits answer
# MISLEAD condition LLMs are instructed to deceive without lying, options are true answer, false answer, misleading answer, none fits answer
# MISLEAD_UNFITTING condition: LLMs are instructed to deceive without lying, options are true answer, false answer, unrelated or nonsense answer, none fits answer

# Test whether LLMs mislead significantly more often in the MISLEAD CONDTION than in the LIE CONDITION -> LLMs can mislead without lying
# Test whether LLMS mislead significantly more often in the MISLEAD CONDITION than in the MISLEAD_UNFITTING CONDITION -> The previous result is not due to LLMs selecting a unrelated non-lie but specifically a deceptive statement