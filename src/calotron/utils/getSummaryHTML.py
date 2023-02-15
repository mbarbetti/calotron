def getSummaryHTML(model) -> tuple:
    headers = ["Layer (type)", "Output shape", "Param #"]
    heads_html = "<tr>\n" + "".join([f"<th>{h}</th>\n" for h in headers]) + "</tr>\n"

    rows = []
    tot_params = 0
    for layer in model.layers:
        layer_type = f"<td>{layer.name} ({layer.__class__.__name__})</td>\n"
        try:
            output_shape = f"<td>{layer.get_output_at(0).get_shape()}</td>\n"
        except RuntimeError:
            output_shape = "<td>None</td>\n"  # print "None" in case of errors
        num_params = f"<td>{layer.count_params()}</td>\n"
        rows.append("<tr>\n" + layer_type + output_shape + num_params + "</tr>\n")
        tot_params += layer.count_params()
    rows_html = "".join([f"{r}" for r in rows])

    table_html = '<table width="40%" border="1px solid black">\n \
                  <thead>\n{}</thead>\n \
                  <tbody>\n{}</tbody>\n \
                  </table>'.format(
        heads_html, rows_html
    )

    return table_html, tot_params
