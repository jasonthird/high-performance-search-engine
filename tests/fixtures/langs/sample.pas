unit Sample;

interface

type
  TWidget = class
    Width: Integer;
    function Render: Integer;
  end;

function ComputeTotal(const Items: array of Integer): Integer;

implementation

function TWidget.Render: Integer;
begin
  Result := Width * 2;
end;

function ComputeTotal(const Items: array of Integer): Integer;
var I: Integer;
begin
  Result := 0;
  for I := Low(Items) to High(Items) do Result := Result + Items[I];
end;

end.
