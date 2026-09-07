package Widget;

sub render {
    my ($self, $width) = @_;
    return " " x $width;
}

sub compute_total {
    my @items = @_;
    my $t = 0;
    $t += $_ for @items;
    return $t;
}
1;
