package body Sample is

   function Compute_Total (Items : Int_Array) return Integer is
      T : Integer := 0;
   begin
      for I in Items'Range loop
         T := T + Items (I);
      end loop;
      return T;
   end Compute_Total;

   procedure Render (Width : Integer) is
   begin
      null;
   end Render;

end Sample;
