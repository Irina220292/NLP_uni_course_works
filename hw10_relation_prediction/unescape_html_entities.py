import glob
import html

HTML_AMPERSAND = "amp;"
HTML_GREATER = "gt;"
HTML_LOWER = "lt;"
HTML_QUOT = "quot;"

# optional, relevance depends on features, model
# none-exhaustive html unescaping

default_path = "data/relations"
for filename in glob.glob(default_path + "/*.tsv"):
    with open(filename.replace(".tsv", "_unescaped.tsv"), "w", encoding="utf-8") as output: # TODO override original to keep original filenames
        with open(filename, encoding='utf-8') as f:
            for line in f:
                for entity in line.split():
                    unescaped_entity = ""
                    if entity.startswith("&"): # support lined up entities (&amp;quot;), but not in-word entities (J&oacute;zef)
                        unescaped_entity = html.unescape(entity).encode("utf-8").decode("utf-8")
                        if unescaped_entity.startswith("&"):
                            entities = entity[1:]
                            unescaped_entity = entities.replace(HTML_AMPERSAND, "&")
                            unescaped_entity = unescaped_entity.replace(HTML_GREATER, ">")
                            unescaped_entity = unescaped_entity.replace(HTML_LOWER, "<")
                            unescaped_entity = unescaped_entity.replace(HTML_QUOT, "\"")
                            # ...

                        line = line.replace(entity, unescaped_entity)
                output.write(line)

