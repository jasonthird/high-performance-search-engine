-module(sample).
-export([compute_total/1, render/1]).

-record(widget, {width = 0}).

compute_total(Items) ->
    lists:sum(Items).

render(Width) ->
    lists:duplicate(Width, $ ).
