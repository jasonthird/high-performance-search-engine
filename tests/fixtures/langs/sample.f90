module sample
  implicit none
contains
  function compute_total(items) result(t)
    integer, intent(in) :: items(:)
    integer :: t
    t = sum(items)
  end function compute_total

  subroutine render(width)
    integer, intent(in) :: width
    print *, repeat(' ', width)
  end subroutine render
end module sample
